import numpy as np

from dlt_analyzer_pro.ai_types import AIConfig, DEFAULT_WEIGHTS
from dlt_analyzer_pro.baseline_guard import blend_with_uniform, fit_baseline_guard
from dlt_analyzer_pro.credible_evaluation import evaluate_zone_credibility
from dlt_analyzer_pro.dynamic_weight import evaluate_dynamic_weights
from dlt_analyzer_pro.importer import load_file
from dlt_analyzer_pro.paths import resource_path


def test_negative_bss_conclusion_direction_is_correct():
    result = evaluate_zone_credibility(
        [0.15] * 20,
        [0.122449] * 20,
        [0] * 20,
        pool_size=35,
        pick_count=5,
        bootstrap_samples=500,
        random_repeats=1000,
        seed=9,
    )
    assert result.brier_skill_score < 0
    assert "高于基线" in result.conclusion
    assert "劣于" in result.conclusion


def test_guard_falls_back_when_model_is_systematically_worse():
    target = np.zeros(35)
    target[:5] = 1.0
    bad = np.full(35, 1e-6)
    bad[-5:] = 1.0
    bad /= bad.sum()
    result = fit_baseline_guard(
        [bad] * 30,
        [target] * 30,
        picks=5,
        bootstrap_samples=500,
        seed=11,
    )
    assert result.model_share == 0.0
    assert result.protected_brier <= result.reference_brier + 1e-12


def test_blend_with_uniform_is_normalized():
    probability = np.arange(1, 36, dtype=float)
    blended = blend_with_uniform(probability, 0.4)
    assert np.isclose(blended.sum(), 1.0)
    assert np.all(blended > 0)


def test_dynamic_window_is_not_silently_clipped_to_twenty():
    draws, failures = load_file(resource_path("dlt_history.csv"))
    assert not failures
    result = evaluate_dynamic_weights(
        draws,
        periods=21,
        current_weights=DEFAULT_WEIGHTS,
        learning_rate=0.2,
        estimators=30,
        include_ml=False,
        seed=17,
    )
    assert result.periods == 21
    assert abs(sum(result.front_weights.values()) - 1.0) < 1e-6
    assert abs(sum(result.back_weights.values()) - 1.0) < 1e-6
    assert 0.0 <= result.front_model_share <= 1.0
    assert 0.0 <= result.back_model_share <= 1.0


def test_zone_specific_config_weights_and_shares():
    config = AIConfig(
        weights=DEFAULT_WEIGHTS,
        front_weights={"bayesian": 1.0},
        back_weights={"markov": 1.0},
        front_model_share=0.2,
        back_model_share=0.7,
    )
    assert config.normalized_zone_weights("front")["bayesian"] == 1.0
    assert config.normalized_zone_weights("back")["markov"] == 1.0
    assert config.zone_model_share("front") == 0.2
    assert config.zone_model_share("back") == 0.7
