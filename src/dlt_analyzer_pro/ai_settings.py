from __future__ import annotations

import json
import os
from pathlib import Path

from .ai_types import DEFAULT_WEIGHTS, DynamicWeightResult
from .paths import app_data_dir


def settings_path() -> Path:
    path = app_data_dir() / "ai_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _defaults() -> dict[str, object]:
    return {
        "weights": dict(DEFAULT_WEIGHTS),
        "front_weights": {},
        "back_weights": {},
        "front_model_share": 1.0,
        "back_model_share": 1.0,
        "baseline_guard": True,
        "simulations": 1_000_000,
        "ga_population": 260,
        "ga_generations": 60,
        "ml_estimators": 140,
        "dynamic_periods": 30,
        "dynamic_learning_rate": 0.20,
        "auto_update": True,
        "deterministic": True,
        "seed": 20260721,
        "probability_calibration": True,
        "model_cache": True,
        "auto_backup": True,
        "backup_retention": 10,
        "leakage_guard": True,
        "last_dynamic_result": None,
    }


def load_ai_settings() -> dict[str, object]:
    defaults = _defaults()
    path = settings_path()
    if not path.exists():
        return defaults
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return defaults
        defaults.update({key: payload[key] for key in defaults if key in payload})
        weights = defaults.get("weights")
        if not isinstance(weights, dict):
            defaults["weights"] = dict(DEFAULT_WEIGHTS)
        else:
            defaults["weights"] = {
                name: float(weights.get(name, value))
                for name, value in DEFAULT_WEIGHTS.items()
            }
        for zone_key in ("front_weights", "back_weights"):
            zone_weights = defaults.get(zone_key)
            if not isinstance(zone_weights, dict):
                defaults[zone_key] = {}
            else:
                defaults[zone_key] = {
                    name: float(zone_weights.get(name, 0.0))
                    for name in DEFAULT_WEIGHTS
                    if name in zone_weights
                }
        defaults["front_model_share"] = float(defaults.get("front_model_share", 1.0))
        defaults["back_model_share"] = float(defaults.get("back_model_share", 1.0))
        defaults["baseline_guard"] = bool(defaults.get("baseline_guard", True))
        return defaults
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return defaults


def save_ai_settings(
    weights: dict[str, float],
    simulations: int,
    ga_population: int,
    ga_generations: int,
    ml_estimators: int,
    dynamic_periods: int | None = None,
    dynamic_learning_rate: float | None = None,
    auto_update: bool | None = None,
    last_dynamic_result: dict | None = None,
    *,
    deterministic: bool | None = None,
    seed: int | None = None,
    probability_calibration: bool | None = None,
    model_cache: bool | None = None,
    auto_backup: bool | None = None,
    backup_retention: int | None = None,
    leakage_guard: bool | None = None,
    front_weights: dict[str, float] | None = None,
    back_weights: dict[str, float] | None = None,
    front_model_share: float | None = None,
    back_model_share: float | None = None,
    baseline_guard: bool | None = None,
) -> None:
    payload = load_ai_settings()
    payload.update(
        {
            "weights": {key: float(value) for key, value in weights.items()},
            "simulations": int(simulations),
            "ga_population": int(ga_population),
            "ga_generations": int(ga_generations),
            "ml_estimators": int(ml_estimators),
        }
    )
    optional = {
        "dynamic_periods": (None if dynamic_periods is None else int(dynamic_periods)),
        "dynamic_learning_rate": (None if dynamic_learning_rate is None else float(dynamic_learning_rate)),
        "auto_update": (None if auto_update is None else bool(auto_update)),
        "deterministic": (None if deterministic is None else bool(deterministic)),
        "seed": (None if seed is None else int(seed)),
        "probability_calibration": (None if probability_calibration is None else bool(probability_calibration)),
        "model_cache": (None if model_cache is None else bool(model_cache)),
        "auto_backup": (None if auto_backup is None else bool(auto_backup)),
        "backup_retention": (None if backup_retention is None else int(backup_retention)),
        "leakage_guard": (None if leakage_guard is None else bool(leakage_guard)),
        "front_model_share": (None if front_model_share is None else float(front_model_share)),
        "back_model_share": (None if back_model_share is None else float(back_model_share)),
        "baseline_guard": (None if baseline_guard is None else bool(baseline_guard)),
    }
    for key, value in optional.items():
        if value is not None:
            payload[key] = value
    if front_weights is not None:
        payload["front_weights"] = {key: float(value) for key, value in front_weights.items()}
    if back_weights is not None:
        payload["back_weights"] = {key: float(value) for key, value in back_weights.items()}
    if last_dynamic_result is not None:
        payload["last_dynamic_result"] = last_dynamic_result
    path = settings_path()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def dynamic_result_to_dict(result: DynamicWeightResult) -> dict:
    return {
        "periods": result.periods,
        "weights": dict(result.weights),
        "confidence": result.confidence,
        "generated_at": result.generated_at,
        "front_weights": dict(result.front_weights),
        "back_weights": dict(result.back_weights),
        "front_model_share": result.front_model_share,
        "back_model_share": result.back_model_share,
        "front_bss": result.front_bss,
        "back_bss": result.back_bss,
        "front_bss_ci": [result.front_bss_ci_lower, result.front_bss_ci_upper],
        "back_bss_ci": [result.back_bss_ci_lower, result.back_bss_ci_upper],
        "guard_notes": list(result.guard_notes),
        "rankings": [
            {
                "model_name": item.model_name,
                "label": item.label,
                "front_brier": item.front_brier,
                "back_brier": item.back_brier,
                "front_hits": item.front_hits,
                "back_hits": item.back_hits,
                "quality_score": item.quality_score,
                "target_weight": item.target_weight,
                "final_weight": item.final_weight,
                "trend": item.trend,
            }
            for item in result.rankings
        ],
    }
