import numpy as np

from dlt_analyzer_pro.ai_backtest import (
    optimize_ensemble_weights,
    walk_forward_ai_backtest,
)
from dlt_analyzer_pro.ai_engine import (
    GeneticOptimizer,
    MonteCarloSimulator,
    learn_structure,
)
from dlt_analyzer_pro.ai_features import recent_five_years
from dlt_analyzer_pro.ai_models import (
    component_scores,
    train_ml_model,
)
from dlt_analyzer_pro.ai_types import AIConfig
from dlt_analyzer_pro.importer import load_file
from dlt_analyzer_pro.paths import resource_path


def history():
    draws, failures = load_file(resource_path("dlt_history.csv"))
    assert not failures
    return draws


def test_recent_five_years_and_statistical_components():
    draws = recent_five_years(history())
    assert len(draws) >= 800
    front, metrics = component_scores(draws, "front", include_ml=False)
    back, _ = component_scores(draws, "back", include_ml=False)
    assert not metrics
    assert set(front) == {
        "bayesian",
        "markov",
        "omission",
        "frequency",
        "xgboost",
        "lightgbm",
    }
    assert all(values.shape == (35,) for values in front.values())
    assert all(values.shape == (12,) for values in back.values())


def test_xgboost_and_lightgbm_feature_scoring():
    draws = history()[-260:]
    xgb = train_ml_model(
        draws,
        "front",
        "xgboost",
        estimators=30,
        fast=True,
    )
    lgb = train_ml_model(
        draws,
        "back",
        "lightgbm",
        estimators=30,
        fast=True,
    )
    assert xgb.current_scores.shape == (35,)
    assert lgb.current_scores.shape == (12,)
    assert xgb.metric.validation_brier is not None
    assert lgb.metric.validation_brier is not None


def test_monte_carlo_runs_at_least_one_million():
    draws = history()
    structure = learn_structure(draws)
    front_probability = np.arange(1, 36, dtype=float)
    back_probability = np.arange(1, 13, dtype=float)
    front_probability /= front_probability.sum()
    back_probability /= back_probability.sum()
    simulator = MonteCarloSimulator(
        front_probability,
        back_probability,
        structure,
        seed=42,
    )
    candidates = simulator.simulate(
        1_000_000,
        batch_size=50_000,
        keep_per_batch=120,
    )
    assert candidates
    assert len(candidates[0][0]) == 5
    assert len(candidates[0][1]) == 2


def test_genetic_optimizer_and_diversity():
    draws = history()
    structure = learn_structure(draws)
    front_probability = np.full(35, 1 / 35)
    back_probability = np.full(12, 1 / 12)
    seeds = [
        ((1, 2, 3, 4, 5), (1, 2), 1.0),
        ((6, 7, 8, 9, 10), (3, 4), 0.9),
    ]
    optimizer = GeneticOptimizer(
        front_probability,
        back_probability,
        structure,
        population_size=80,
        generations=20,
        seed=42,
    )
    ranked = optimizer.optimize(seeds)
    assert len(ranked) >= 20
    assert len(ranked[0][0]) == 5
    assert len(ranked[0][1]) == 2


def test_ai_backtest_and_parameter_optimization_without_ml():
    draws = history()
    config = AIConfig(
        simulations=1_000_000,
        ga_population=80,
        ga_generations=20,
        ml_estimators=30,
    )
    result = walk_forward_ai_backtest(
        draws,
        periods=3,
        config=config,
        include_ml=False,
    )
    assert result.evaluated == 3
    optimized = optimize_ensemble_weights(
        draws,
        periods=3,
        trials=20,
        config=config,
        include_ml=False,
    )
    assert optimized.evaluated_periods == 3
    assert abs(sum(optimized.best_weights.values()) - 1.0) < 1e-6
