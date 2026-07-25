from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ZoneCredibleEvaluation:
    model_brier: float
    reference_brier: float
    brier_skill_score: float
    bss_ci_lower: float
    bss_ci_upper: float
    bss_probability_positive: float
    model_hit_average: float
    model_hit_ci_lower: float
    model_hit_ci_upper: float
    random_hit_average: float
    random_hit_ci_lower: float
    random_hit_ci_upper: float
    hit_uplift: float
    random_p_value: float
    conclusion: str


def brier_skill_score(model_brier: float, reference_brier: float) -> float:
    """BSS = 1 - BS_model / BS_reference. Positive values are better."""
    denominator = max(float(reference_brier), 1e-15)
    return float(1.0 - float(model_brier) / denominator)


def _percentile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = (1.0 - float(confidence_level)) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)



def bootstrap_bss_interval(
    model_briers: Sequence[float],
    reference_briers: Sequence[float],
    *,
    bootstrap_samples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 20260721,
) -> tuple[float, float]:
    model_loss = np.asarray(model_briers, dtype=float)
    reference_loss = np.asarray(reference_briers, dtype=float)
    if model_loss.ndim != 1 or len(model_loss) == 0:
        raise ValueError("model_briers must contain at least one value")
    if len(reference_loss) != len(model_loss):
        raise ValueError("model and reference Brier sequences must have the same length")
    bootstrap_samples = max(500, int(bootstrap_samples))
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(model_loss), size=(bootstrap_samples, len(model_loss)))
    boot_model = np.mean(model_loss[indices], axis=1)
    boot_reference = np.maximum(np.mean(reference_loss[indices], axis=1), 1e-15)
    return _percentile_interval(1.0 - boot_model / boot_reference, confidence_level)

def evaluate_zone_credibility(
    model_briers: Sequence[float],
    reference_briers: Sequence[float],
    model_hits: Sequence[float],
    *,
    pool_size: int,
    pick_count: int,
    bootstrap_samples: int = 2_000,
    random_repeats: int = 5_000,
    confidence_level: float = 0.95,
    seed: int = 20260721,
) -> ZoneCredibleEvaluation:
    model_loss = np.asarray(model_briers, dtype=float)
    reference_loss = np.asarray(reference_briers, dtype=float)
    hits = np.asarray(model_hits, dtype=float)
    if model_loss.ndim != 1 or len(model_loss) == 0:
        raise ValueError('model_briers must contain at least one value')
    if len(reference_loss) != len(model_loss) or len(hits) != len(model_loss):
        raise ValueError('all evaluation sequences must have the same length')
    if not (0.5 < confidence_level < 1.0):
        raise ValueError('confidence_level must be between 0.5 and 1.0')

    bootstrap_samples = max(500, int(bootstrap_samples))
    random_repeats = max(1_000, int(random_repeats))
    rng = np.random.default_rng(int(seed))
    periods = len(model_loss)

    model_brier = float(np.mean(model_loss))
    reference_brier = float(np.mean(reference_loss))
    skill = brier_skill_score(model_brier, reference_brier)

    indices = rng.integers(0, periods, size=(bootstrap_samples, periods))
    boot_model = np.mean(model_loss[indices], axis=1)
    boot_reference = np.maximum(np.mean(reference_loss[indices], axis=1), 1e-15)
    boot_skill = 1.0 - boot_model / boot_reference
    bss_lower, bss_upper = _percentile_interval(boot_skill, confidence_level)
    probability_positive = float(np.mean(boot_skill > 0.0))

    boot_hits = np.mean(hits[indices], axis=1)
    hit_lower, hit_upper = _percentile_interval(boot_hits, confidence_level)
    model_hit_average = float(np.mean(hits))

    # Exact repeated random-ticket experiment via the equivalent hypergeometric law.
    # Each repeat contains the same number of periods as the model backtest.
    random_hits = rng.hypergeometric(
        ngood=int(pick_count),
        nbad=int(pool_size - pick_count),
        nsample=int(pick_count),
        size=(random_repeats, periods),
    )
    random_means = np.mean(random_hits, axis=1)
    random_average = float(np.mean(random_means))
    random_lower, random_upper = _percentile_interval(random_means, confidence_level)
    p_value = float((1 + np.count_nonzero(random_means >= model_hit_average)) / (random_repeats + 1))
    uplift = float(model_hit_average - random_average)

    if bss_lower > 0.0:
        conclusion = 'Brier显著优于基线'
    elif bss_upper < 0.0:
        conclusion = 'Brier显著高于基线，模型劣于均匀概率基线'
    elif uplift > 0.0 and p_value < 0.05:
        conclusion = '命中高于随机，但Brier未显著'
    else:
        conclusion = '未证实稳定优势'

    return ZoneCredibleEvaluation(
        model_brier=model_brier,
        reference_brier=reference_brier,
        brier_skill_score=skill,
        bss_ci_lower=bss_lower,
        bss_ci_upper=bss_upper,
        bss_probability_positive=probability_positive,
        model_hit_average=model_hit_average,
        model_hit_ci_lower=hit_lower,
        model_hit_ci_upper=hit_upper,
        random_hit_average=random_average,
        random_hit_ci_lower=random_lower,
        random_hit_ci_upper=random_upper,
        hit_uplift=uplift,
        random_p_value=p_value,
        conclusion=conclusion,
    )
