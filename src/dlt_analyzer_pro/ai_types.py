from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .credible_evaluation import ZoneCredibleEvaluation


COMPONENT_NAMES = (
    "bayesian",
    "markov",
    "omission",
    "frequency",
    "xgboost",
    "lightgbm",
)

COMPONENT_LABELS = {
    "bayesian": "贝叶斯概率更新",
    "markov": "马尔可夫链",
    "omission": "遗漏周期模型",
    "frequency": "近期频率",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

DEFAULT_WEIGHTS = {
    "bayesian": 0.16,
    "markov": 0.11,
    "omission": 0.14,
    "frequency": 0.09,
    "xgboost": 0.25,
    "lightgbm": 0.25,
}


@dataclass(frozen=True, slots=True)
class AIConfig:
    simulations: int = 1_000_000
    ga_population: int = 260
    ga_generations: int = 60
    prediction_count: int = 10
    seed: int = 20260721
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    front_weights: Mapping[str, float] = field(default_factory=dict)
    back_weights: Mapping[str, float] = field(default_factory=dict)
    front_model_share: float = 1.0
    back_model_share: float = 1.0
    baseline_guard: bool = True
    ml_estimators: int = 140
    dynamic_periods: int = 30
    dynamic_learning_rate: float = 0.20
    auto_update: bool = True
    deterministic: bool = True
    probability_calibration: bool = True
    model_cache: bool = True
    auto_backup: bool = True
    backup_retention: int = 10
    leakage_guard: bool = True

    @staticmethod
    def _normalize_mapping(values: Mapping[str, float]) -> dict[str, float]:
        cleaned = {
            name: max(0.0, float(values.get(name, 0.0)))
            for name in COMPONENT_NAMES
        }
        total = sum(cleaned.values())
        if total <= 0:
            return dict(DEFAULT_WEIGHTS)
        return {key: value / total for key, value in cleaned.items()}

    def normalized_weights(self) -> dict[str, float]:
        return self._normalize_mapping(self.weights)

    def normalized_zone_weights(self, zone: str) -> dict[str, float]:
        if zone == "front" and self.front_weights:
            return self._normalize_mapping(self.front_weights)
        if zone == "back" and self.back_weights:
            return self._normalize_mapping(self.back_weights)
        return self.normalized_weights()

    def zone_model_share(self, zone: str) -> float:
        value = self.front_model_share if zone == "front" else self.back_model_share
        return min(1.0, max(0.0, float(value))) if self.baseline_guard else 1.0


@dataclass(frozen=True, slots=True)
class ModelMetric:
    model_name: str
    zone: str
    train_rows: int
    validation_brier: float | None
    validation_auc: float | None
    uncalibrated_brier: float | None = None
    calibration_method: str = "identity"
    model_version: str = ""
    cache_status: str = "trained"


@dataclass(frozen=True, slots=True)
class ModelPerformance:
    model_name: str
    label: str
    front_brier: float
    back_brier: float
    front_hits: float
    back_hits: float
    quality_score: float
    target_weight: float
    final_weight: float
    trend: float


@dataclass(frozen=True, slots=True)
class DynamicWeightResult:
    periods: int
    weights: Mapping[str, float]
    rankings: tuple[ModelPerformance, ...]
    confidence: float
    generated_at: str
    front_weights: Mapping[str, float] = field(default_factory=dict)
    back_weights: Mapping[str, float] = field(default_factory=dict)
    front_model_share: float = 1.0
    back_model_share: float = 1.0
    front_bss: float = 0.0
    back_bss: float = 0.0
    front_bss_ci_lower: float = 0.0
    front_bss_ci_upper: float = 0.0
    back_bss_ci_lower: float = 0.0
    back_bss_ci_upper: float = 0.0
    guard_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AIReport:
    dataset_count: int
    simulations: int
    elapsed_seconds: float
    component_weights: Mapping[str, float]
    model_metrics: tuple[ModelMetric, ...]
    front_scores: Mapping[int, float]
    back_scores: Mapping[int, float]
    weight_source: str = "manual"
    dynamic_result: DynamicWeightResult | None = None
    deterministic_seed: int = 0
    dataset_fingerprint: str = ""
    leakage_audit_passed: bool = True
    stability_notes: tuple[str, ...] = ()
    front_component_weights: Mapping[str, float] = field(default_factory=dict)
    back_component_weights: Mapping[str, float] = field(default_factory=dict)
    front_model_share: float = 1.0
    back_model_share: float = 1.0
    baseline_guard_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AIBacktestResult:
    evaluated: int
    model_front_average: float
    model_back_average: float
    random_front_average: float
    random_back_average: float
    front_brier: float
    back_brier: float
    details: tuple[dict[str, object], ...]
    front_evaluation: ZoneCredibleEvaluation | None = None
    back_evaluation: ZoneCredibleEvaluation | None = None
    bootstrap_samples: int = 0
    random_repeats: int = 0
    confidence_level: float = 0.95


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    evaluated_periods: int
    trials: int
    best_weights: Mapping[str, float]
    best_objective: float
    front_brier: float
    back_brier: float
    front_hits: float
    back_hits: float
