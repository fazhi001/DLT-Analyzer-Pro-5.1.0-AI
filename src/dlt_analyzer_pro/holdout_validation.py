from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    holdout_periods: int
    baseline_loss: float
    candidate_loss: float
    skill_score: float
    ci_lower: float
    ci_upper: float
    enabled: bool
    reason: str


def evaluate_final_holdout(
    candidate_losses: Sequence[float],
    baseline_losses: Sequence[float],
    *,
    minimum_holdout: int = 30,
    seed: int = 20260802,
) -> HoldoutResult:
    """Evaluate a frozen model on periods excluded from model selection."""
    candidate = np.asarray(candidate_losses, dtype=float)
    baseline = np.asarray(baseline_losses, dtype=float)
    if candidate.ndim != 1 or len(candidate) != len(baseline):
        raise ValueError("candidate and baseline losses must be aligned")
    holdout = max(10, int(minimum_holdout))
    if len(candidate) < holdout:
        raise ValueError("not enough independent holdout periods")
    candidate = candidate[-holdout:]
    baseline = baseline[-holdout:]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, holdout, size=(2000, holdout))
    skill_samples = 1.0 - candidate[indices].mean(axis=1) / np.maximum(
        baseline[indices].mean(axis=1), 1e-15
    )
    candidate_mean = float(candidate.mean())
    baseline_mean = float(baseline.mean())
    skill = 1.0 - candidate_mean / max(baseline_mean, 1e-15)
    lower, upper = (float(x) for x in np.quantile(skill_samples, [0.025, 0.975]))
    enabled = bool(skill > 0.0 and lower > 0.0)
    return HoldoutResult(
        holdout_periods=holdout,
        baseline_loss=baseline_mean,
        candidate_loss=candidate_mean,
        skill_score=float(skill),
        ci_lower=lower,
        ci_upper=upper,
        enabled=enabled,
        reason=(
            "最终留出集Brier显著优于基线，允许启用AI"
            if enabled else "最终留出集未证实稳定优势，已停用AI"
        ),
    )
