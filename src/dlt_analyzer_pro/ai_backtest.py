from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Callable

import numpy as np

from .ai_engine import _weighted_ensemble
from .credible_evaluation import evaluate_zone_credibility
from .baseline_guard import blend_with_uniform
from .ai_features import recent_five_years
from .ai_models import component_scores
from .ai_types import (
    AIBacktestResult,
    AIConfig,
    OptimizationResult,
)
from .models import Draw


ProgressCallback = Callable[[str, float], None]


def _notify(callback: ProgressCallback | None, text: str, value: float) -> None:
    if callback is not None:
        callback(text, max(0.0, min(1.0, float(value))))


def _target_vector(draw: Draw, zone: str) -> np.ndarray:
    pool = 35 if zone == "front" else 12
    values = np.zeros(pool, dtype=float)
    numbers = draw.front if zone == "front" else draw.back
    values[np.asarray(numbers, dtype=int) - 1] = 1.0
    return values


def _top_numbers(probability: np.ndarray, count: int) -> tuple[int, ...]:
    indices = np.argpartition(probability, -count)[-count:]
    return tuple(sorted(int(index) + 1 for index in indices))


def _brier(probability: np.ndarray, target: np.ndarray, picks: int) -> float:
    scaled = probability * picks
    return float(np.mean((scaled - target) ** 2))


def walk_forward_ai_backtest(
    draws: list[Draw],
    periods: int = 30,
    config: AIConfig | None = None,
    include_ml: bool = True,
    progress: ProgressCallback | None = None,
    *,
    bootstrap_samples: int = 2_000,
    random_repeats: int = 5_000,
    confidence_level: float = 0.95,
) -> AIBacktestResult:
    config = config or AIConfig()
    selected = recent_five_years(draws)
    periods = max(3, min(int(periods), len(selected) - 80))
    bootstrap_samples = max(500, int(bootstrap_samples))
    random_repeats = max(1_000, int(random_repeats))
    start = len(selected) - periods
    rng = random.Random(config.seed)
    details: list[dict[str, object]] = []
    front_briers: list[float] = []
    back_briers: list[float] = []
    front_reference_briers: list[float] = []
    back_reference_briers: list[float] = []
    model_front_hits: list[int] = []
    model_back_hits: list[int] = []

    front_reference_probability = np.full(35, 1.0 / 35.0, dtype=float)
    back_reference_probability = np.full(12, 1.0 / 12.0, dtype=float)

    for offset, index in enumerate(range(start, len(selected))):
        training = selected[:index]
        actual = selected[index]
        front_components, _ = component_scores(
            training,
            "front",
            estimators=max(36, config.ml_estimators // 3),
            random_state=config.seed + index,
            include_ml=include_ml,
            fast_ml=True,
            calibrate=config.probability_calibration,
        )
        back_components, _ = component_scores(
            training,
            "back",
            estimators=max(36, config.ml_estimators // 3),
            random_state=config.seed + index + 1,
            include_ml=include_ml,
            fast_ml=True,
            calibrate=config.probability_calibration,
        )
        front_probability = _weighted_ensemble(
            front_components, config.normalized_zone_weights("front")
        )
        back_probability = _weighted_ensemble(
            back_components, config.normalized_zone_weights("back")
        )
        front_probability = blend_with_uniform(
            front_probability, config.zone_model_share("front")
        )
        back_probability = blend_with_uniform(
            back_probability, config.zone_model_share("back")
        )
        predicted_front = _top_numbers(front_probability, 5)
        predicted_back = _top_numbers(back_probability, 2)
        random_front = tuple(sorted(rng.sample(range(1, 36), 5)))
        random_back = tuple(sorted(rng.sample(range(1, 13), 2)))

        target_front = _target_vector(actual, "front")
        target_back = _target_vector(actual, "back")
        front_loss = _brier(front_probability, target_front, 5)
        back_loss = _brier(back_probability, target_back, 2)
        front_reference_loss = _brier(front_reference_probability, target_front, 5)
        back_reference_loss = _brier(back_reference_probability, target_back, 2)
        front_hit = len(set(predicted_front) & set(actual.front))
        back_hit = len(set(predicted_back) & set(actual.back))

        front_briers.append(front_loss)
        back_briers.append(back_loss)
        front_reference_briers.append(front_reference_loss)
        back_reference_briers.append(back_reference_loss)
        model_front_hits.append(front_hit)
        model_back_hits.append(back_hit)

        details.append(
            {
                "issue": actual.issue,
                "model_front_hits": front_hit,
                "model_back_hits": back_hit,
                "random_front_hits": len(set(random_front) & set(actual.front)),
                "random_back_hits": len(set(random_back) & set(actual.back)),
                "model_front_brier": front_loss,
                "model_back_brier": back_loss,
                "reference_front_brier": front_reference_loss,
                "reference_back_brier": back_reference_loss,
            }
        )
        _notify(
            progress,
            f"滚动样本外回测 {offset + 1}/{periods}",
            0.82 * (offset + 1) / periods,
        )

    _notify(progress, f"Bootstrap置信区间 {bootstrap_samples:,}次", 0.88)
    front_evaluation = evaluate_zone_credibility(
        front_briers,
        front_reference_briers,
        model_front_hits,
        pool_size=35,
        pick_count=5,
        bootstrap_samples=bootstrap_samples,
        random_repeats=random_repeats,
        confidence_level=confidence_level,
        seed=config.seed + 70_001,
    )
    _notify(progress, f"随机基线重复实验 {random_repeats:,}次", 0.94)
    back_evaluation = evaluate_zone_credibility(
        back_briers,
        back_reference_briers,
        model_back_hits,
        pool_size=12,
        pick_count=2,
        bootstrap_samples=bootstrap_samples,
        random_repeats=random_repeats,
        confidence_level=confidence_level,
        seed=config.seed + 70_002,
    )
    _notify(progress, "可信评估完成", 1.0)

    return AIBacktestResult(
        evaluated=periods,
        model_front_average=float(np.mean(model_front_hits)),
        model_back_average=float(np.mean(model_back_hits)),
        random_front_average=front_evaluation.random_hit_average,
        random_back_average=back_evaluation.random_hit_average,
        front_brier=front_evaluation.model_brier,
        back_brier=back_evaluation.model_brier,
        details=tuple(details),
        front_evaluation=front_evaluation,
        back_evaluation=back_evaluation,
        bootstrap_samples=bootstrap_samples,
        random_repeats=random_repeats,
        confidence_level=confidence_level,
    )


def optimize_ensemble_weights(
    draws: list[Draw],
    periods: int = 10,
    trials: int = 80,
    config: AIConfig | None = None,
    include_ml: bool = True,
    progress: ProgressCallback | None = None,
) -> OptimizationResult:
    config = config or AIConfig()
    selected = recent_five_years(draws)
    periods = max(3, min(int(periods), len(selected) - 80))
    trials = max(20, int(trials))
    start = len(selected) - periods
    component_names = (
        "bayesian",
        "markov",
        "omission",
        "frequency",
        "xgboost",
        "lightgbm",
    )
    folds: list[dict[str, object]] = []

    for offset, index in enumerate(range(start, len(selected))):
        training = selected[:index]
        actual = selected[index]
        front_components, _ = component_scores(
            training,
            "front",
            estimators=max(30, config.ml_estimators // 4),
            random_state=config.seed + index,
            include_ml=include_ml,
            fast_ml=True,
            calibrate=config.probability_calibration,
        )
        back_components, _ = component_scores(
            training,
            "back",
            estimators=max(30, config.ml_estimators // 4),
            random_state=config.seed + index + 1,
            include_ml=include_ml,
            fast_ml=True,
            calibrate=config.probability_calibration,
        )
        folds.append(
            {
                "front": front_components,
                "back": back_components,
                "actual": actual,
            }
        )
        _notify(
            progress,
            f"预计算优化样本 {offset + 1}/{periods}",
            0.45 * (offset + 1) / periods,
        )

    rng = np.random.default_rng(config.seed)
    candidates = [
        np.asarray(
            [config.normalized_weights()[name] for name in component_names],
            dtype=float,
        )
    ]
    for _ in range(trials - 1):
        candidates.append(rng.dirichlet(np.ones(len(component_names)) * 1.35))

    best_objective = -1e18
    best_weights: dict[str, float] = {}
    best_values = (math.inf, math.inf, 0.0, 0.0)

    for trial_index, vector in enumerate(candidates):
        weights = {
            name: float(vector[index])
            for index, name in enumerate(component_names)
        }
        front_briers: list[float] = []
        back_briers: list[float] = []
        front_hits: list[int] = []
        back_hits: list[int] = []

        for fold in folds:
            actual = fold["actual"]
            front_probability = _weighted_ensemble(fold["front"], weights)
            back_probability = _weighted_ensemble(fold["back"], weights)
            target_front = _target_vector(actual, "front")
            target_back = _target_vector(actual, "back")
            front_briers.append(_brier(front_probability, target_front, 5))
            back_briers.append(_brier(back_probability, target_back, 2))
            front_hits.append(
                len(set(_top_numbers(front_probability, 5)) & set(actual.front))
            )
            back_hits.append(
                len(set(_top_numbers(back_probability, 2)) & set(actual.back))
            )

        mean_front_brier = float(np.mean(front_briers))
        mean_back_brier = float(np.mean(back_briers))
        mean_front_hits = float(np.mean(front_hits))
        mean_back_hits = float(np.mean(back_hits))
        # Optimize a strictly proper out-of-sample probability objective.
        # Hit counts are reported for interpretation but do not drive the weights.
        concentration_penalty = 0.01 * float(np.sum(np.square(vector)))
        objective = -(
            (5.0 / 7.0) * mean_front_brier
            + (2.0 / 7.0) * mean_back_brier
            + concentration_penalty
        )
        if objective > best_objective:
            best_objective = objective
            best_weights = weights
            best_values = (
                mean_front_brier,
                mean_back_brier,
                mean_front_hits,
                mean_back_hits,
            )
        _notify(
            progress,
            f"参数优化 {trial_index + 1}/{trials}",
            0.45 + 0.55 * (trial_index + 1) / trials,
        )

    return OptimizationResult(
        evaluated_periods=periods,
        trials=trials,
        best_weights=best_weights,
        best_objective=float(best_objective),
        front_brier=best_values[0],
        back_brier=best_values[1],
        front_hits=best_values[2],
        back_hits=best_values[3],
    )
