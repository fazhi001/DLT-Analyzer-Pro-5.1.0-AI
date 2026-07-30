from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from dlt_analyzer_pro.digit_model import (
    DigitPredictionEngine,
    _blend_probability,
    _enumerated_digits,
    enumerate_digit_candidates,
)
from dlt_analyzer_pro.models import DigitDraw


def make_draws(game: str, count: int = 150) -> list[DigitDraw]:
    positions = 3 if game == "pl3" else 5
    start = date(2025, 1, 1)
    return [
        DigitDraw(
            game=game,
            issue=str(26000 + index),
            draw_date=start + timedelta(days=index),
            digits=tuple(
                (index * (position + 2) + position * 3 + index // 7) % 10
                for position in range(positions)
            ),
        )
        for index in range(count)
    ]


def test_complete_enumeration_sizes_and_boundaries():
    pl3 = _enumerated_digits(3)
    pl5 = _enumerated_digits(5)
    assert pl3.shape == (1000, 3)
    assert pl5.shape == (100000, 5)
    assert tuple(pl3[0]) == (0, 0, 0)
    assert tuple(pl3[-1]) == (9, 9, 9)
    assert tuple(pl5[0]) == (0, 0, 0, 0, 0)
    assert tuple(pl5[-1]) == (9, 9, 9, 9, 9)


def test_enumeration_ranks_known_peak_first():
    probabilities = []
    expected = (1, 2, 3)
    for preferred in expected:
        probability = np.full(10, 0.01)
        probability[preferred] = 0.91
        probability /= probability.sum()
        probabilities.append(probability)
    digits, scores = enumerate_digit_candidates(
        probabilities,
        historical_sums=np.asarray([6.0] * 30),
        strategy="均衡模式",
    )
    assert tuple(digits[0]) == expected
    assert scores[0] >= scores[1]


def test_blend_probability_is_normalized():
    baseline = np.full((4, 10), 0.1)
    model = np.full((4, 10), 0.02)
    model[:, 7] = 0.82
    blended = _blend_probability(baseline, model, 0.35)
    assert np.allclose(blended.sum(axis=1), 1.0)
    assert np.all(blended > 0)


def test_generation_is_deterministic_across_seeds(tmp_path):
    draws = make_draws("pl5", 140)
    first = DigitPredictionEngine(
        "pl5", seed=1, model_base_dir=tmp_path / "models-a"
    ).generate(draws, count=20, use_ml=False)
    second = DigitPredictionEngine(
        "pl5", seed=999, model_base_dir=tmp_path / "models-b"
    ).generate(draws, count=20, use_ml=False)
    assert [item.digits for item in first] == [item.digits for item in second]
    assert len({item.digits for item in first}) == 20


def test_training_uses_bounded_dynamic_weights(tmp_path):
    draws = make_draws("pl3", 150)
    engine = DigitPredictionEngine("pl3", model_base_dir=tmp_path / "models")
    report = engine.train_models(draws, force=True)
    assert len(report.statuses) == 3
    assert all(0.0 <= status.ml_weight <= 0.70 for status in report.statuses)
    assert all(status.validation_periods >= 20 for status in report.statuses)
    assert all(0.0 <= status.fold_win_rate <= 1.0 for status in report.statuses)
