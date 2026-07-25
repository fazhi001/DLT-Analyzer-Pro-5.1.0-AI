from __future__ import annotations

import sys

import numpy as np

from dlt_analyzer_pro.ai_models import train_ml_model
from dlt_analyzer_pro.crash_reporting import write_crash_report
from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.importer import load_file
from dlt_analyzer_pro.model_registry import ModelRegistry
from dlt_analyzer_pro.paths import resource_path
from dlt_analyzer_pro.probability_calibration import fit_probability_calibrator
from dlt_analyzer_pro.stability import audit_training_pipeline, set_global_seed


def history():
    draws, failures = load_file(resource_path("dlt_history.csv"))
    assert not failures
    return draws


def test_temporal_leakage_audit_passes_reference_history():
    result = audit_training_pipeline(history())
    assert result.passed
    assert result.critical_count == 0
    assert len(result.fingerprint) == 64


def test_temporal_leakage_audit_detects_duplicate_issue():
    draws = history()
    broken = draws[:120] + [draws[119]]
    result = audit_training_pipeline(broken)
    assert not result.passed
    assert any(item.code == "duplicate_issue" for item in result.issues)


def test_probability_calibration_never_worsens_held_out_brier():
    rng = np.random.default_rng(42)
    raw = np.clip(rng.beta(2, 5, size=800), 1e-5, 1 - 1e-5)
    target = rng.binomial(1, np.clip(raw * 0.72 + 0.04, 0, 1))
    calibrator, calibrated, raw_brier, calibrated_brier = fit_probability_calibrator(
        raw, target, seed=42, enabled=True
    )
    assert calibrated.shape == raw.shape
    assert calibrated_brier <= raw_brier + 1e-8
    assert calibrator.method in {"identity", "platt"}


def test_model_registry_cache_and_rollback(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    metadata = {
        "fingerprint": "a" * 64,
        "created_at": "2026-07-22T10:00:00",
        "latest_issue": "26081",
        "validation_brier": 0.2,
        "calibrated_brier": 0.19,
        "validation_auc": 0.51,
        "seed": 42,
        "estimators": 30,
    }
    version = registry.register(
        model_name="xgboost",
        zone="front",
        bundle={"estimator": {"value": 1}},
        metadata=metadata,
    )
    loaded = registry.load_for_prediction("xgboost", "front", "a" * 64)
    assert loaded is not None
    assert loaded[2] == "cache-hit"
    registry.pin_version(version)
    pinned = registry.load_for_prediction("xgboost", "front", "b" * 64)
    assert pinned is not None
    assert pinned[2] == "rollback"
    registry.unpin()
    assert not any(item.pinned for item in registry.list_versions())


def test_ml_model_registry_integration(tmp_path):
    draws = history()[-180:]
    registry = ModelRegistry(tmp_path / "models")
    first = train_ml_model(
        draws,
        "front",
        "xgboost",
        estimators=24,
        random_state=123,
        fast=False,
        registry=registry,
        use_registry=True,
        persist_registry=True,
    )
    second = train_ml_model(
        draws,
        "front",
        "xgboost",
        estimators=24,
        random_state=123,
        fast=False,
        registry=registry,
        use_registry=True,
        persist_registry=True,
    )
    assert first.metric.model_version
    assert second.metric.cache_status == "cache-hit"
    assert np.allclose(first.current_scores, second.current_scores)


def test_verified_database_backup_and_interval(tmp_path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    database.upsert_draws(history()[:20])
    backup = database.verified_backup(tmp_path / "backups", retention=2)
    assert backup.exists()
    assert database.automatic_backup(
        tmp_path / "backups", retention=2, minimum_interval_hours=24
    ) is None
    healthy, detail = database.integrity_check()
    assert healthy and detail == "ok"


def test_seed_and_crash_report(tmp_path):
    assert set_global_seed(1234) == 1234
    try:
        raise RuntimeError("stability-test")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
        path = write_crash_report(
            exc_type, exc_value, exc_tb, context="pytest", data_dir=tmp_path
        )
    assert path.exists()
    assert "stability-test" in path.read_text(encoding="utf-8")
