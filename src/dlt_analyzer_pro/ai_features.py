from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from .models import Draw


FEATURE_NAMES = (
    "number_scaled",
    "last_present",
    "freq_5",
    "freq_10",
    "freq_20",
    "freq_50",
    "freq_100",
    "trend_10_50",
    "omission_scaled",
    "mean_gap_scaled",
    "gap_std_scaled",
    "markov_probability",
    "bayesian_probability",
    "recency_weighted_frequency",
    "zone_scaled",
    "previous_sum_scaled",
    "time_sin",
    "time_cos",
)


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    X: np.ndarray
    y: np.ndarray
    time_index: np.ndarray
    current_X: np.ndarray
    numbers: np.ndarray
    feature_names: tuple[str, ...]


def recent_five_years(draws: list[Draw]) -> list[Draw]:
    if not draws:
        return []
    dated = [draw for draw in draws if draw.draw_date is not None]
    if len(dated) >= 200:
        latest_date = max(draw.draw_date for draw in dated if draw.draw_date is not None)
        cutoff = latest_date - timedelta(days=5 * 366)
        selected = [
            draw
            for draw in draws
            if draw.draw_date is not None and draw.draw_date >= cutoff
        ]
        if len(selected) >= 300:
            return selected
    # 大乐透每年约150期；日期尚未补齐时按最近850期兜底。
    return draws[-min(len(draws), 850):]


def presence_matrix(draws: list[Draw], zone: str) -> np.ndarray:
    pool = 35 if zone == "front" else 12
    matrix = np.zeros((len(draws), pool), dtype=np.uint8)
    for row, draw in enumerate(draws):
        numbers = draw.front if zone == "front" else draw.back
        matrix[row, np.asarray(numbers, dtype=int) - 1] = 1
    return matrix


def _omission(history: np.ndarray) -> int:
    positions = np.flatnonzero(history)
    if positions.size == 0:
        return int(history.size)
    return int(history.size - 1 - positions[-1])


def _gap_stats(history: np.ndarray) -> tuple[float, float]:
    positions = np.flatnonzero(history)
    if positions.size < 2:
        return float(max(1, history.size)), 0.0
    gaps = np.diff(positions).astype(float)
    return float(gaps.mean()), float(gaps.std())


def _markov_probability(history: np.ndarray) -> float:
    if history.size < 3:
        return float(history.mean()) if history.size else 0.0
    previous = history[:-1]
    following = history[1:]
    current_state = int(history[-1])
    mask = previous == current_state
    successes = int(following[mask].sum())
    trials = int(mask.sum())
    return (successes + 1.0) / (trials + 2.0)


def _bayesian_probability(history: np.ndarray, expected_rate: float) -> float:
    if history.size == 0:
        return expected_rate
    age = np.arange(history.size - 1, -1, -1, dtype=float)
    weights = np.power(0.992, age)
    prior_strength = 18.0
    alpha = expected_rate * prior_strength
    beta = (1.0 - expected_rate) * prior_strength
    successes = float(np.dot(history, weights))
    trials = float(weights.sum())
    return (alpha + successes) / (alpha + beta + trials)


def _recency_frequency(history: np.ndarray) -> float:
    if history.size == 0:
        return 0.0
    age = np.arange(history.size - 1, -1, -1, dtype=float)
    weights = np.power(0.975, age)
    return float(np.dot(history, weights) / weights.sum())


def _feature_row(
    matrix: np.ndarray,
    t: int,
    number_index: int,
    zone: str,
    previous_sum: float,
) -> list[float]:
    history = matrix[:t, number_index].astype(float)
    pool = matrix.shape[1]
    picks = 5 if zone == "front" else 2
    expected_rate = picks / pool

    def freq(window: int) -> float:
        values = history[-window:]
        return float(values.mean()) if values.size else expected_rate

    omission = _omission(history)
    mean_gap, gap_std = _gap_stats(history)
    zone_value = (
        (number_index + 1) / pool
        if zone == "back"
        else (0.0 if number_index < 12 else 0.5 if number_index < 24 else 1.0)
    )
    phase = 2.0 * np.pi * t / 156.0
    return [
        (number_index + 1) / pool,
        float(history[-1]) if history.size else 0.0,
        freq(5),
        freq(10),
        freq(20),
        freq(50),
        freq(100),
        freq(10) - freq(50),
        min(omission / 40.0, 2.0),
        min(mean_gap / 40.0, 2.0),
        min(gap_std / 30.0, 2.0),
        _markov_probability(history),
        _bayesian_probability(history, expected_rate),
        _recency_frequency(history),
        zone_value,
        previous_sum / (175.0 if zone == "front" else 24.0),
        float(np.sin(phase)),
        float(np.cos(phase)),
    ]


def build_feature_dataset(
    draws: list[Draw],
    zone: str,
    min_history: int = 35,
) -> FeatureDataset:
    if zone not in {"front", "back"}:
        raise ValueError("zone must be front or back")
    selected = recent_five_years(draws)
    if len(selected) <= min_history:
        raise ValueError("历史数据不足以训练机器学习模型")

    matrix = presence_matrix(selected, zone)
    pool = matrix.shape[1]
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    time_rows: list[int] = []

    for t in range(min_history, len(selected)):
        previous_draw = selected[t - 1]
        previous_sum = (
            sum(previous_draw.front) if zone == "front" else sum(previous_draw.back)
        )
        for number_index in range(pool):
            X_rows.append(
                _feature_row(matrix, t, number_index, zone, previous_sum)
            )
            y_rows.append(int(matrix[t, number_index]))
            time_rows.append(t)

    previous_draw = selected[-1]
    previous_sum = (
        sum(previous_draw.front) if zone == "front" else sum(previous_draw.back)
    )
    current_X = np.asarray(
        [
            _feature_row(matrix, len(selected), number_index, zone, previous_sum)
            for number_index in range(pool)
        ],
        dtype=np.float32,
    )

    return FeatureDataset(
        X=np.asarray(X_rows, dtype=np.float32),
        y=np.asarray(y_rows, dtype=np.uint8),
        time_index=np.asarray(time_rows, dtype=np.int32),
        current_X=current_X,
        numbers=np.arange(1, pool + 1, dtype=np.int16),
        feature_names=FEATURE_NAMES,
    )
