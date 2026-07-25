from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "DLTAnalyzerPro2"


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath("resources", *parts)


def app_data_dir(override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_DIR_NAME


def ensure_app_dirs(override: Path | None = None) -> Path:
    base = app_data_dir(override)
    for name in ("", "logs", "exports", "backups", "models", "crash_reports", "stability"):
        (base / name).mkdir(parents=True, exist_ok=True)
    return base


def database_path(override: Path | None = None) -> Path:
    return ensure_app_dirs(override) / "dlt_analyzer_v2.db"


def log_path(override: Path | None = None) -> Path:
    return ensure_app_dirs(override) / "logs" / "app.log"


def model_dir(override: Path | None = None) -> Path:
    return ensure_app_dirs(override) / "models"


def crash_report_dir(override: Path | None = None) -> Path:
    return ensure_app_dirs(override) / "crash_reports"


def stability_report_path(override: Path | None = None) -> Path:
    return ensure_app_dirs(override) / "stability" / "latest_audit.json"
