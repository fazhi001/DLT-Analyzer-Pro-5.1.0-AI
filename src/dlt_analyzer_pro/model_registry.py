from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from .paths import app_data_dir
from .stability import atomic_write_json


@dataclass(frozen=True, slots=True)
class ModelVersionInfo:
    version: str
    model_name: str
    zone: str
    created_at: str
    latest_issue: str
    fingerprint: str
    validation_brier: float | None
    validation_auc: float | None
    calibrated_brier: float | None
    seed: int
    estimators: int
    active: bool
    pinned: bool
    path: str


class ModelRegistry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or (app_data_dir() / "models"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.quarantine_dir = self.root / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _empty_manifest(self) -> dict[str, object]:
        return {"schema": 1, "versions": [], "active": {}, "pinned": {}}

    def _read_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return self._empty_manifest()
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._empty_manifest()
            for key, default in self._empty_manifest().items():
                payload.setdefault(key, default)
            return payload
        except (OSError, ValueError, json.JSONDecodeError):
            return self._empty_manifest()

    @staticmethod
    def _key(model_name: str, zone: str) -> str:
        return f"{model_name}:{zone}"

    def register(
        self,
        *,
        model_name: str,
        zone: str,
        bundle: dict[str, Any],
        metadata: dict[str, object],
        retention: int = 6,
    ) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fingerprint = str(metadata["fingerprint"])
        version = f"{model_name}_{zone}_{stamp}_{fingerprint[:8]}"
        folder = self.root / model_name / zone
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{version}.joblib"
        temporary = target.with_suffix(".joblib.tmp")
        metadata = dict(metadata)
        metadata["version"] = version
        joblib.dump({"bundle": bundle, "metadata": metadata}, temporary)
        os.replace(temporary, target)

        manifest = self._read_manifest()
        versions = list(manifest.get("versions", []))
        entry = {
            "version": version,
            "model_name": model_name,
            "zone": zone,
            "created_at": metadata.get("created_at", datetime.now().isoformat(timespec="seconds")),
            "latest_issue": metadata.get("latest_issue", ""),
            "fingerprint": fingerprint,
            "validation_brier": metadata.get("validation_brier"),
            "validation_auc": metadata.get("validation_auc"),
            "calibrated_brier": metadata.get("calibrated_brier"),
            "seed": metadata.get("seed", 0),
            "estimators": metadata.get("estimators", 0),
            "path": str(target.relative_to(self.root)),
        }
        versions.append(entry)
        manifest["versions"] = versions
        key = self._key(model_name, zone)
        active = dict(manifest.get("active", {}))
        active[key] = version
        manifest["active"] = active
        atomic_write_json(self.manifest_path, manifest)
        self._prune(model_name, zone, retention)
        return version

    def _prune(self, model_name: str, zone: str, retention: int) -> None:
        manifest = self._read_manifest()
        key = self._key(model_name, zone)
        pinned = str(dict(manifest.get("pinned", {})).get(key, ""))
        matching = [
            entry for entry in manifest.get("versions", [])
            if entry.get("model_name") == model_name and entry.get("zone") == zone
        ]
        matching.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        keep_versions = {str(item["version"]) for item in matching[: max(2, int(retention))]}
        if pinned:
            keep_versions.add(pinned)
        removed: set[str] = set()
        for entry in matching:
            version = str(entry["version"])
            if version in keep_versions:
                continue
            path = self.root / str(entry["path"])
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            removed.add(version)
        if removed:
            manifest["versions"] = [
                entry for entry in manifest.get("versions", [])
                if str(entry.get("version")) not in removed
            ]
            atomic_write_json(self.manifest_path, manifest)

    def load_for_prediction(
        self,
        model_name: str,
        zone: str,
        fingerprint: str,
    ) -> tuple[dict[str, Any], dict[str, object], str] | None:
        manifest = self._read_manifest()
        key = self._key(model_name, zone)
        pinned = str(dict(manifest.get("pinned", {})).get(key, ""))
        versions = list(manifest.get("versions", []))
        candidates = []
        if pinned:
            candidates.extend(entry for entry in versions if entry.get("version") == pinned)
        candidates.extend(
            entry for entry in reversed(versions)
            if entry.get("model_name") == model_name
            and entry.get("zone") == zone
            and entry.get("fingerprint") == fingerprint
            and entry.get("version") != pinned
        )
        for entry in candidates:
            path = self.root / str(entry["path"])
            try:
                payload = joblib.load(path)
                status = "rollback" if str(entry.get("version")) == pinned and pinned else "cache-hit"
                return payload["bundle"], payload["metadata"], status
            except Exception:
                self._quarantine(entry, path)
        return None

    def _quarantine(self, entry: dict[str, object], path: Path) -> None:
        try:
            if path.exists():
                target = self.quarantine_dir / path.name
                shutil.move(str(path), str(target))
        except OSError:
            pass
        manifest = self._read_manifest()
        manifest["versions"] = [
            item for item in manifest.get("versions", [])
            if item.get("version") != entry.get("version")
        ]
        atomic_write_json(self.manifest_path, manifest)

    def list_versions(self) -> list[ModelVersionInfo]:
        manifest = self._read_manifest()
        active = dict(manifest.get("active", {}))
        pinned = dict(manifest.get("pinned", {}))
        rows: list[ModelVersionInfo] = []
        for entry in manifest.get("versions", []):
            key = self._key(str(entry.get("model_name")), str(entry.get("zone")))
            rows.append(
                ModelVersionInfo(
                    version=str(entry.get("version")),
                    model_name=str(entry.get("model_name")),
                    zone=str(entry.get("zone")),
                    created_at=str(entry.get("created_at")),
                    latest_issue=str(entry.get("latest_issue", "")),
                    fingerprint=str(entry.get("fingerprint", "")),
                    validation_brier=(None if entry.get("validation_brier") is None else float(entry["validation_brier"])),
                    validation_auc=(None if entry.get("validation_auc") is None else float(entry["validation_auc"])),
                    calibrated_brier=(None if entry.get("calibrated_brier") is None else float(entry["calibrated_brier"])),
                    seed=int(entry.get("seed", 0)),
                    estimators=int(entry.get("estimators", 0)),
                    active=str(active.get(key, "")) == str(entry.get("version")),
                    pinned=str(pinned.get(key, "")) == str(entry.get("version")),
                    path=str(entry.get("path", "")),
                )
            )
        rows.sort(key=lambda item: item.created_at, reverse=True)
        return rows

    def pin_version(self, version: str) -> None:
        manifest = self._read_manifest()
        entry = next(
            (item for item in manifest.get("versions", []) if item.get("version") == version),
            None,
        )
        if entry is None:
            raise ValueError("找不到所选模型版本")
        key = self._key(str(entry["model_name"]), str(entry["zone"]))
        pinned = dict(manifest.get("pinned", {}))
        pinned[key] = version
        manifest["pinned"] = pinned
        atomic_write_json(self.manifest_path, manifest)

    def unpin(self, model_name: str | None = None, zone: str | None = None) -> None:
        manifest = self._read_manifest()
        if model_name and zone:
            key = self._key(model_name, zone)
            pinned = dict(manifest.get("pinned", {}))
            pinned.pop(key, None)
            manifest["pinned"] = pinned
        else:
            manifest["pinned"] = {}
        atomic_write_json(self.manifest_path, manifest)
