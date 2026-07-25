from __future__ import annotations

import random

from .models import BacktestDetail, BacktestResult, Draw
from .predictor import PredictionEngine


def rolling_backtest(
    draws: list[Draw],
    periods: int = 50,
    strategy: str = "均衡模式",
    seed: int = 20260721,
) -> BacktestResult:
    if len(draws) < 60:
        raise ValueError("至少需要60期数据进行前向验证")

    start = max(40, len(draws) - max(1, int(periods)))
    random_engine = random.Random(seed)
    details: list[BacktestDetail] = []

    for index in range(start, len(draws)):
        training = draws[:index]
        actual = draws[index]
        prediction = PredictionEngine(seed + index).generate(
            training,
            count=1,
            strategy=strategy,
            candidate_count=600,
        )[0]

        random_front = tuple(sorted(random_engine.sample(range(1, 36), 5)))
        random_back = tuple(sorted(random_engine.sample(range(1, 13), 2)))

        details.append(
            BacktestDetail(
                issue=actual.issue,
                model_front_hits=len(set(prediction.front) & set(actual.front)),
                model_back_hits=len(set(prediction.back) & set(actual.back)),
                random_front_hits=len(set(random_front) & set(actual.front)),
                random_back_hits=len(set(random_back) & set(actual.back)),
            )
        )

    count = len(details)
    return BacktestResult(
        evaluated=count,
        model_front_average=sum(d.model_front_hits for d in details) / count,
        model_back_average=sum(d.model_back_hits for d in details) / count,
        random_front_average=sum(d.random_front_hits for d in details) / count,
        random_back_average=sum(d.random_back_hits for d in details) / count,
        details=tuple(details),
    )
