from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrator:
    method: str = "identity"
    slope: float = 1.0
    intercept: float = 0.0

    def transform(self, probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.method != "platt":
            return values
        logit = np.log(values / (1.0 - values))
        calibrated = 1.0 / (1.0 + np.exp(-(self.slope * logit + self.intercept)))
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


def fit_probability_calibrator(
    validation_probability: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    enabled: bool = True,
) -> tuple[ProbabilityCalibrator, np.ndarray, float, float]:
    raw = np.clip(np.asarray(validation_probability, dtype=float), 1e-6, 1 - 1e-6)
    target = np.asarray(y_validation, dtype=int)
    raw_brier = float(brier_score_loss(target, raw))
    if not enabled or len(raw) < 100 or np.unique(target).size < 2:
        return ProbabilityCalibrator(), raw, raw_brier, raw_brier

    logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    model = LogisticRegression(
        C=10.0,
        solver="lbfgs",
        max_iter=500,
        random_state=int(seed),
    )
    model.fit(logit, target)
    candidate = ProbabilityCalibrator(
        method="platt",
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )
    calibrated = candidate.transform(raw)
    calibrated_brier = float(brier_score_loss(target, calibrated))

    # Stability guard: do not keep calibration when it degrades the held-out Brier.
    if calibrated_brier > raw_brier + 1e-8:
        return ProbabilityCalibrator(), raw, raw_brier, raw_brier
    return candidate, calibrated, raw_brier, calibrated_brier
