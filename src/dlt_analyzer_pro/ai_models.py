from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score

from .ai_features import build_feature_dataset, presence_matrix, recent_five_years
from .ai_types import ModelMetric
from .model_registry import ModelRegistry
from .models import Draw
from .probability_calibration import (
    ProbabilityCalibrator,
    fit_probability_calibrator,
)
from .stability import training_fingerprint


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.full_like(values, 0.5, dtype=float)
    return (values - low) / (high - low)


def bayesian_scores(draws: list[Draw], zone: str) -> np.ndarray:
    selected = recent_five_years(draws)
    matrix = presence_matrix(selected, zone).astype(float)
    pool = matrix.shape[1]
    picks = 5 if zone == "front" else 2
    expected_rate = picks / pool
    age = np.arange(len(selected) - 1, -1, -1, dtype=float)
    weights = np.power(0.992, age)
    prior_strength = 20.0
    alpha = expected_rate * prior_strength
    beta = (1.0 - expected_rate) * prior_strength
    successes = weights @ matrix
    posterior = (alpha + successes) / (alpha + beta + weights.sum())
    return _normalize(posterior)


def markov_scores(draws: list[Draw], zone: str) -> np.ndarray:
    selected = recent_five_years(draws)
    matrix = presence_matrix(selected, zone)
    output = np.zeros(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        series = matrix[:, index]
        current = int(series[-1])
        previous = series[:-1]
        following = series[1:]
        mask = previous == current
        output[index] = (float(following[mask].sum()) + 1.0) / (
            float(mask.sum()) + 2.0
        )
    return _normalize(output)


def omission_cycle_scores(draws: list[Draw], zone: str) -> np.ndarray:
    selected = recent_five_years(draws)
    matrix = presence_matrix(selected, zone)
    output = np.zeros(matrix.shape[1], dtype=float)
    for index in range(matrix.shape[1]):
        positions = np.flatnonzero(matrix[:, index])
        if positions.size == 0:
            output[index] = 0.5
            continue
        gaps = np.diff(positions).astype(int)
        current_omission = len(selected) - 1 - int(positions[-1])
        target_gap = current_omission + 1
        if gaps.size == 0:
            output[index] = min(1.0, target_gap / 20.0)
            continue
        at_risk = max(1, int(np.sum(gaps >= target_gap)))
        near_events = int(np.sum(np.abs(gaps - target_gap) <= 1))
        hazard = (near_events + 1.0) / (at_risk + 3.0)
        mean_gap = max(1.0, float(gaps.mean()))
        overdue = 1.0 / (1.0 + np.exp(-(target_gap / mean_gap - 1.0) * 2.0))
        output[index] = 0.62 * hazard + 0.38 * overdue
    return _normalize(output)


def frequency_scores(draws: list[Draw], zone: str) -> np.ndarray:
    selected = recent_five_years(draws)
    matrix = presence_matrix(selected, zone).astype(float)
    age = np.arange(len(selected) - 1, -1, -1, dtype=float)
    weights = np.power(0.985, age)
    recent = (weights @ matrix) / weights.sum()
    short = matrix[-30:].mean(axis=0)
    return _normalize(0.55 * recent + 0.45 * short)


@dataclass(slots=True)
class TrainedMLModel:
    name: str
    zone: str
    estimator: object
    metric: ModelMetric
    current_scores: np.ndarray
    calibrator: ProbabilityCalibrator


def _build_estimator(
    model_name: str,
    count: int,
    scale_pos_weight: float,
    random_state: int,
    fast: bool,
):
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=count,
            max_depth=3,
            learning_rate=0.055 if not fast else 0.08,
            min_child_weight=3,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.08,
            reg_lambda=1.25,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            random_state=random_state,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=count,
            num_leaves=15,
            max_depth=-1,
            learning_rate=0.055 if not fast else 0.08,
            min_child_samples=24,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.08,
            reg_lambda=1.25,
            class_weight="balanced",
            verbosity=-1,
            n_jobs=2,
            random_state=random_state,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"未知机器学习模型：{model_name}")


def _metric_from_metadata(
    metadata: dict[str, object],
    *,
    model_name: str,
    zone: str,
    cache_status: str,
) -> ModelMetric:
    return ModelMetric(
        model_name=model_name,
        zone="前区" if zone == "front" else "后区",
        train_rows=int(metadata.get("train_rows", 0)),
        validation_brier=(
            None
            if metadata.get("validation_brier") is None
            else float(metadata["validation_brier"])
        ),
        validation_auc=(
            None
            if metadata.get("validation_auc") is None
            else float(metadata["validation_auc"])
        ),
        uncalibrated_brier=(
            None
            if metadata.get("uncalibrated_brier") is None
            else float(metadata["uncalibrated_brier"])
        ),
        calibration_method=str(metadata.get("calibration_method", "identity")),
        model_version=str(metadata.get("version", "")),
        cache_status=cache_status,
    )


def train_ml_model(
    draws: list[Draw],
    zone: str,
    model_name: str,
    estimators: int = 140,
    random_state: int = 20260721,
    fast: bool = False,
    *,
    calibrate: bool = True,
    registry: ModelRegistry | None = None,
    use_registry: bool = False,
    persist_registry: bool = False,
) -> TrainedMLModel:
    dataset = build_feature_dataset(draws, zone)
    count = max(24, int(estimators // 3)) if fast else int(estimators)
    fingerprint = training_fingerprint(
        draws,
        zone,
        model_name,
        count,
        random_state,
        calibration=calibrate,
    )

    model_registry = registry or (ModelRegistry() if use_registry or persist_registry else None)
    if use_registry and not fast and model_registry is not None:
        loaded = model_registry.load_for_prediction(model_name, zone, fingerprint)
        if loaded is not None:
            bundle, metadata, cache_status = loaded
            estimator = bundle["estimator"]
            calibrator = bundle.get("calibrator", ProbabilityCalibrator())
            try:
                current_probability = calibrator.transform(
                    estimator.predict_proba(dataset.current_X)[:, 1]
                )
            except Exception:
                # A pinned model from an incompatible feature schema is not
                # allowed to break prediction. Return to automatic training.
                model_registry.unpin(model_name, zone)
            else:
                return TrainedMLModel(
                    name=model_name,
                    zone=zone,
                    estimator=estimator,
                    metric=_metric_from_metadata(
                        metadata,
                        model_name=model_name,
                        zone=zone,
                        cache_status=cache_status,
                    ),
                    current_scores=_normalize(current_probability),
                    calibrator=calibrator,
                )

    unique_times = np.unique(dataset.time_index)
    validation_time = unique_times[max(1, int(len(unique_times) * 0.84))]
    train_mask = dataset.time_index < validation_time
    validation_mask = ~train_mask

    X_train = dataset.X[train_mask]
    y_train = dataset.y[train_mask]
    X_validation = dataset.X[validation_mask]
    y_validation = dataset.y[validation_mask]

    positive = max(1, int(y_train.sum()))
    negative = max(1, int(len(y_train) - positive))
    scale_pos_weight = negative / positive

    validation_estimator = _build_estimator(
        model_name,
        count,
        scale_pos_weight,
        random_state,
        fast,
    )
    validation_estimator.fit(X_train, y_train)
    raw_validation_probability = validation_estimator.predict_proba(X_validation)[:, 1]
    calibrator, validation_probability, raw_brier, calibrated_brier = (
        fit_probability_calibrator(
            raw_validation_probability,
            y_validation,
            seed=random_state,
            enabled=calibrate,
        )
    )

    auc = None
    if np.unique(y_validation).size == 2:
        auc = float(roc_auc_score(y_validation, validation_probability))

    # Final model uses all available historical labels. Fast rolling folds retain
    # the validation estimator to avoid doubling backtest cost.
    if fast:
        estimator = validation_estimator
    else:
        all_positive = max(1, int(dataset.y.sum()))
        all_negative = max(1, int(len(dataset.y) - all_positive))
        estimator = _build_estimator(
            model_name,
            count,
            all_negative / all_positive,
            random_state,
            False,
        )
        estimator.fit(dataset.X, dataset.y)

    current_probability = calibrator.transform(
        estimator.predict_proba(dataset.current_X)[:, 1]
    )

    metadata: dict[str, object] = {
        "model_name": model_name,
        "zone": zone,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "latest_issue": str(draws[-1].issue if draws else ""),
        "fingerprint": fingerprint,
        "train_rows": int(len(dataset.X) if not fast else len(X_train)),
        "validation_brier": calibrated_brier,
        "uncalibrated_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "validation_auc": auc,
        "calibration_method": calibrator.method,
        "seed": int(random_state),
        "estimators": int(count),
    }
    version = ""
    if persist_registry and not fast and model_registry is not None:
        version = model_registry.register(
            model_name=model_name,
            zone=zone,
            bundle={"estimator": estimator, "calibrator": calibrator},
            metadata=metadata,
        )
        metadata["version"] = version
    metric = _metric_from_metadata(
        metadata,
        model_name=model_name,
        zone=zone,
        cache_status="trained",
    )
    if version:
        metric = ModelMetric(
            model_name=metric.model_name,
            zone=metric.zone,
            train_rows=metric.train_rows,
            validation_brier=metric.validation_brier,
            validation_auc=metric.validation_auc,
            uncalibrated_brier=metric.uncalibrated_brier,
            calibration_method=metric.calibration_method,
            model_version=version,
            cache_status="trained",
        )
    return TrainedMLModel(
        name=model_name,
        zone=zone,
        estimator=estimator,
        metric=metric,
        current_scores=_normalize(current_probability),
        calibrator=calibrator,
    )


def component_scores(
    draws: list[Draw],
    zone: str,
    estimators: int = 140,
    random_state: int = 20260721,
    include_ml: bool = True,
    fast_ml: bool = False,
    *,
    calibrate: bool = True,
    registry: ModelRegistry | None = None,
    use_registry: bool = False,
    persist_registry: bool = False,
) -> tuple[dict[str, np.ndarray], tuple[ModelMetric, ...]]:
    components = {
        "bayesian": bayesian_scores(draws, zone),
        "markov": markov_scores(draws, zone),
        "omission": omission_cycle_scores(draws, zone),
        "frequency": frequency_scores(draws, zone),
    }
    metrics: list[ModelMetric] = []
    if include_ml:
        for model_name in ("xgboost", "lightgbm"):
            trained = train_ml_model(
                draws,
                zone,
                model_name,
                estimators=estimators,
                random_state=random_state,
                fast=fast_ml,
                calibrate=calibrate,
                registry=registry,
                use_registry=use_registry,
                persist_registry=persist_registry,
            )
            components[model_name] = trained.current_scores
            metrics.append(trained.metric)
    else:
        components["xgboost"] = components["frequency"].copy()
        components["lightgbm"] = components["frequency"].copy()
    return components, tuple(metrics)
