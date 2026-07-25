from __future__ import annotations

from datetime import datetime

import numpy as np

from .ai_engine import _weighted_ensemble
from .ai_features import recent_five_years
from .ai_models import component_scores
from .ai_types import (
    COMPONENT_LABELS,
    COMPONENT_NAMES,
    DEFAULT_WEIGHTS,
    DynamicWeightResult,
    ModelPerformance,
)
from .baseline_guard import fit_baseline_guard
from .models import Draw


def _probability(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=float), 1e-9)
    output /= output.sum()
    return output


def _target(draw: Draw, zone: str) -> np.ndarray:
    pool = 35 if zone == "front" else 12
    output = np.zeros(pool, dtype=float)
    numbers = draw.front if zone == "front" else draw.back
    output[np.asarray(numbers, dtype=int) - 1] = 1.0
    return output


def _hits(probability: np.ndarray, actual: tuple[int, ...], picks: int) -> int:
    indices = np.argpartition(probability, -picks)[-picks:]
    predicted = {int(index) + 1 for index in indices}
    return len(predicted & set(actual))


def _brier(probability: np.ndarray, target: np.ndarray, picks: int) -> float:
    return float(np.mean((probability * picks - target) ** 2))


def _normalize_quality(values: np.ndarray, inverse: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-12:
        return np.full(values.shape, 0.5, dtype=float)
    normalized = (values - low) / (high - low)
    return 1.0 - normalized if inverse else normalized


def _bounded_weights(raw: np.ndarray, floor: float = 0.025, ceiling: float = 0.50) -> np.ndarray:
    weights = np.asarray(raw, dtype=float)
    weights = np.maximum(weights, 1e-12)
    weights /= weights.sum()
    for _ in range(12):
        weights = np.clip(weights, floor, ceiling)
        weights /= weights.sum()
    return weights


def _current_vector(current_weights: dict[str, float] | None) -> np.ndarray:
    current = current_weights or DEFAULT_WEIGHTS
    vector = np.asarray(
        [max(0.0, float(current.get(name, 0.0))) for name in COMPONENT_NAMES],
        dtype=float,
    )
    if vector.sum() <= 0:
        vector = np.asarray([DEFAULT_WEIGHTS[name] for name in COMPONENT_NAMES], dtype=float)
    vector /= vector.sum()
    return vector


def _zone_weight_vector(
    rows: list[dict[str, float]],
    zone: str,
    current: np.ndarray,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    losses = np.asarray([row[f"{zone}_brier"] for row in rows], dtype=float)
    picks = 5.0 if zone == "front" else 2.0
    hit_rates = np.asarray([row[f"{zone}_hits"] / picks for row in rows], dtype=float)

    # Primary objective is strictly proper probability quality. Hit rate only breaks
    # near-ties and cannot dominate the Brier loss.
    loss_quality = _normalize_quality(losses, inverse=True)
    hit_quality = _normalize_quality(hit_rates)
    quality = 0.90 * loss_quality + 0.10 * hit_quality

    temperature = 2.0
    target_raw = np.exp(temperature * (quality - quality.max()))
    target = _bounded_weights(target_raw)
    rate = min(0.60, max(0.05, float(learning_rate)))
    final = _bounded_weights((1.0 - rate) * current + rate * target)
    return quality, target, final


def evaluate_dynamic_weights(
    draws: list[Draw],
    periods: int = 30,
    current_weights: dict[str, float] | None = None,
    learning_rate: float = 0.20,
    estimators: int = 140,
    include_ml: bool = True,
    calibrate: bool = True,
    seed: int = 20260721,
    progress=None,
) -> DynamicWeightResult:
    """Rank components by rolling OOS Brier and apply a uniform-baseline guard."""
    selected = recent_five_years(draws)
    if len(selected) < 100:
        raise ValueError("历史数据不足，无法进行动态权重评估")

    # 4.2.2 silently clipped this value to 20. The guard version accepts a real
    # 30/50/100-period window, constrained only by available history.
    periods = max(3, min(int(periods), 100, len(selected) - 80))
    start = len(selected) - periods
    metrics = {
        name: {"front_brier": [], "back_brier": [], "front_hits": [], "back_hits": []}
        for name in COMPONENT_NAMES
    }
    front_folds: list[dict[str, np.ndarray]] = []
    back_folds: list[dict[str, np.ndarray]] = []
    front_targets: list[np.ndarray] = []
    back_targets: list[np.ndarray] = []

    for offset, index in enumerate(range(start, len(selected))):
        training = selected[:index]
        actual = selected[index]
        front_components, _ = component_scores(
            training,
            "front",
            estimators=max(30, min(int(estimators), 120) // 2),
            random_state=seed + index,
            include_ml=include_ml,
            fast_ml=True,
            calibrate=calibrate,
        )
        back_components, _ = component_scores(
            training,
            "back",
            estimators=max(30, min(int(estimators), 120) // 2),
            random_state=seed + index + 1,
            include_ml=include_ml,
            fast_ml=True,
            calibrate=calibrate,
        )
        front_components = {name: _probability(front_components[name]) for name in COMPONENT_NAMES}
        back_components = {name: _probability(back_components[name]) for name in COMPONENT_NAMES}
        target_front = _target(actual, "front")
        target_back = _target(actual, "back")

        front_folds.append(front_components)
        back_folds.append(back_components)
        front_targets.append(target_front)
        back_targets.append(target_back)

        for name in COMPONENT_NAMES:
            front_probability = front_components[name]
            back_probability = back_components[name]
            metrics[name]["front_brier"].append(_brier(front_probability, target_front, 5))
            metrics[name]["back_brier"].append(_brier(back_probability, target_back, 2))
            metrics[name]["front_hits"].append(_hits(front_probability, actual.front, 5))
            metrics[name]["back_hits"].append(_hits(back_probability, actual.back, 2))
        if progress is not None:
            progress(f"滚动评估模型 {offset + 1}/{periods}", 0.72 * (offset + 1) / periods)

    rows: list[dict[str, float | str]] = []
    for name in COMPONENT_NAMES:
        rows.append(
            {
                "name": name,
                "front_brier": float(np.mean(metrics[name]["front_brier"])),
                "back_brier": float(np.mean(metrics[name]["back_brier"])),
                "front_hits": float(np.mean(metrics[name]["front_hits"])),
                "back_hits": float(np.mean(metrics[name]["back_hits"])),
            }
        )

    current = _current_vector(current_weights)
    front_quality, front_target, front_final = _zone_weight_vector(rows, "front", current, learning_rate)
    back_quality, back_target, back_final = _zone_weight_vector(rows, "back", current, learning_rate)
    front_weights = {name: float(front_final[index]) for index, name in enumerate(COMPONENT_NAMES)}
    back_weights = {name: float(back_final[index]) for index, name in enumerate(COMPONENT_NAMES)}

    front_probabilities = [_weighted_ensemble(fold, front_weights) for fold in front_folds]
    back_probabilities = [_weighted_ensemble(fold, back_weights) for fold in back_folds]
    if progress is not None:
        progress("计算前后区基线保护", 0.82)
    front_guard = fit_baseline_guard(
        front_probabilities,
        front_targets,
        picks=5,
        bootstrap_samples=max(1000, min(5000, periods * 100)),
        seed=seed + 81_001,
    )
    back_guard = fit_baseline_guard(
        back_probabilities,
        back_targets,
        picks=2,
        bootstrap_samples=max(1000, min(5000, periods * 100)),
        seed=seed + 81_002,
    )

    # The six weights shown in the legacy UI remain a front/back weighted summary.
    combined_final = (5.0 / 7.0) * front_final + (2.0 / 7.0) * back_final
    combined_final /= combined_final.sum()
    combined_target = (5.0 / 7.0) * front_target + (2.0 / 7.0) * back_target
    combined_target /= combined_target.sum()
    combined_quality = (5.0 / 7.0) * front_quality + (2.0 / 7.0) * back_quality

    performances = []
    for index, row in enumerate(rows):
        performances.append(
            ModelPerformance(
                model_name=str(row["name"]),
                label=COMPONENT_LABELS[str(row["name"])],
                front_brier=float(row["front_brier"]),
                back_brier=float(row["back_brier"]),
                front_hits=float(row["front_hits"]),
                back_hits=float(row["back_hits"]),
                quality_score=float(100.0 * combined_quality[index]),
                target_weight=float(combined_target[index]),
                final_weight=float(combined_final[index]),
                trend=float(combined_final[index] - current[index]),
            )
        )
    performances.sort(key=lambda item: (item.final_weight, item.quality_score), reverse=True)
    if progress is not None:
        progress("动态权重与基线保护完成", 1.0)

    return DynamicWeightResult(
        periods=periods,
        weights={name: float(combined_final[index]) for index, name in enumerate(COMPONENT_NAMES)},
        rankings=tuple(performances),
        confidence=float(1.0 - np.exp(-periods / 20.0)),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        front_weights=front_weights,
        back_weights=back_weights,
        front_model_share=front_guard.model_share,
        back_model_share=back_guard.model_share,
        front_bss=front_guard.raw_bss,
        back_bss=back_guard.raw_bss,
        front_bss_ci_lower=front_guard.bss_ci_lower,
        front_bss_ci_upper=front_guard.bss_ci_upper,
        back_bss_ci_lower=back_guard.bss_ci_lower,
        back_bss_ci_upper=back_guard.bss_ci_upper,
        guard_notes=(front_guard.reason, back_guard.reason),
    )
