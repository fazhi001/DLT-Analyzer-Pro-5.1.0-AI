from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .credible_evaluation import brier_skill_score, bootstrap_bss_interval


@dataclass(frozen=True, slots=True)
class BaselineGuardResult:
    model_share: float
    baseline_share: float
    model_brier: float
    protected_brier: float
    reference_brier: float
    raw_bss: float
    protected_bss: float
    bss_ci_lower: float
    bss_ci_upper: float
    reason: str


def blend_with_uniform(probability: np.ndarray, model_share: float) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probability must be a non-empty one-dimensional array")
    values = np.maximum(values, 1e-12)
    values /= values.sum()
    share = float(np.clip(model_share, 0.0, 1.0))
    uniform = np.full(values.size, 1.0 / values.size, dtype=float)
    blended = share * values + (1.0 - share) * uniform
    blended /= blended.sum()
    return blended


def _brier(probability: np.ndarray, target: np.ndarray, picks: int) -> float:
    return float(np.mean((np.asarray(probability) * picks - np.asarray(target)) ** 2))


def _optimal_model_share(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    picks: int,
) -> float:
    if not probabilities:
        return 0.0
    pool = int(np.asarray(probabilities[0]).size)
    uniform_scaled = np.full(pool, picks / pool, dtype=float)
    numerator = 0.0
    denominator = 0.0
    for probability, target in zip(probabilities, targets, strict=True):
        model_scaled = np.asarray(probability, dtype=float) * picks
        direction = model_scaled - uniform_scaled
        residual = uniform_scaled - np.asarray(target, dtype=float)
        numerator += float(np.sum(direction * residual))
        denominator += float(np.sum(direction * direction))
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip(-numerator / denominator, 0.0, 1.0))


def fit_baseline_guard(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    picks: int,
    bootstrap_samples: int = 2_000,
    seed: int = 20260721,
) -> BaselineGuardResult:
    if len(probabilities) == 0 or len(probabilities) != len(targets):
        raise ValueError("probabilities and targets must have the same non-zero length")

    normalized = []
    for probability in probabilities:
        values = np.maximum(np.asarray(probability, dtype=float), 1e-12)
        values /= values.sum()
        normalized.append(values)

    pool = normalized[0].size
    reference_probability = np.full(pool, 1.0 / pool, dtype=float)
    model_losses = np.asarray(
        [_brier(probability, target, picks) for probability, target in zip(normalized, targets, strict=True)],
        dtype=float,
    )
    reference_losses = np.asarray(
        [_brier(reference_probability, target, picks) for target in targets],
        dtype=float,
    )
    raw_model_brier = float(model_losses.mean())
    reference_brier = float(reference_losses.mean())
    raw_bss = brier_skill_score(raw_model_brier, reference_brier)
    ci_lower, ci_upper = bootstrap_bss_interval(
        model_losses,
        reference_losses,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    optimal_share = _optimal_model_share(normalized, targets, picks)
    if ci_upper < 0.0:
        model_share = 0.0
        reason = "样本外BSS置信区间完全低于0，已回退均匀概率基线"
    elif raw_bss <= 0.0:
        model_share = min(optimal_share, 0.25)
        reason = "样本外BSS不高于0，模型概率仅保留最多25%"
    elif ci_lower <= 0.0:
        model_share = min(optimal_share, 0.50)
        reason = "优势尚不显著，模型概率向均匀基线收缩至少50%"
    else:
        model_share = optimal_share
        reason = "样本外BSS显著为正，按验证集最优比例融合基线"

    protected = [blend_with_uniform(probability, model_share) for probability in normalized]
    protected_losses = np.asarray(
        [_brier(probability, target, picks) for probability, target in zip(protected, targets, strict=True)],
        dtype=float,
    )
    protected_brier = float(protected_losses.mean())
    protected_bss = brier_skill_score(protected_brier, reference_brier)

    # Numerical safety: never keep a protected blend that is worse than the pure baseline
    # on the guard validation window.
    if protected_brier > reference_brier + 1e-12:
        model_share = 0.0
        protected_brier = reference_brier
        protected_bss = 0.0
        reason = "保护后Brier仍高于基线，已强制回退均匀概率基线"

    return BaselineGuardResult(
        model_share=float(model_share),
        baseline_share=float(1.0 - model_share),
        model_brier=raw_model_brier,
        protected_brier=protected_brier,
        reference_brier=reference_brier,
        raw_bss=raw_bss,
        protected_bss=protected_bss,
        bss_ci_lower=ci_lower,
        bss_ci_upper=ci_upper,
        reason=reason,
    )
