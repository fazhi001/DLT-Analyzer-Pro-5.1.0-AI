from __future__ import annotations

import random

from .digit_model import DigitPredictionEngine, GAME_POSITIONS
from .models import DigitBacktestDetail, DigitBacktestResult, DigitDraw


def rolling_digit_backtest(
    draws: list[DigitDraw],
    periods: int = 50,
    strategy: str = "均衡模式",
    seed: int = 20260724,
    use_ml: bool = False,
) -> DigitBacktestResult:
    if not draws:
        raise ValueError("没有可用于回测的数据")
    game = draws[-1].game
    expected = GAME_POSITIONS[game]
    valid = [draw for draw in draws if draw.game == game and len(draw.digits) == expected]
    if len(valid) < 60:
        raise ValueError("至少需要60期数据进行排列玩法滚动验证")

    start = max(40, len(valid) - max(1, min(int(periods), 200)))
    random_engine = random.Random(seed)
    details: list[DigitBacktestDetail] = []
    model_position_hits = [0] * expected
    random_position_hits = [0] * expected

    for index in range(start, len(valid)):
        training = valid[:index]
        actual = valid[index]
        prediction = DigitPredictionEngine(game, seed + index).generate(
            training,
            count=1,
            strategy=strategy,
            candidate_count=800,
            use_ml=use_ml,
        )[0]
        random_digits = tuple(random_engine.randrange(10) for _ in range(expected))
        model_flags = [prediction.digits[pos] == actual.digits[pos] for pos in range(expected)]
        random_flags = [random_digits[pos] == actual.digits[pos] for pos in range(expected)]
        for pos in range(expected):
            model_position_hits[pos] += int(model_flags[pos])
            random_position_hits[pos] += int(random_flags[pos])
        details.append(
            DigitBacktestDetail(
                issue=actual.issue,
                model_hits=sum(model_flags),
                random_hits=sum(random_flags),
                exact_model=all(model_flags),
                exact_random=all(random_flags),
            )
        )

    count = len(details)
    return DigitBacktestResult(
        game=game,
        evaluated=count,
        model_average_hits=sum(item.model_hits for item in details) / count,
        random_average_hits=sum(item.random_hits for item in details) / count,
        model_exact_hits=sum(item.exact_model for item in details),
        random_exact_hits=sum(item.exact_random for item in details),
        position_model_rates=tuple(value / count for value in model_position_hits),
        position_random_rates=tuple(value / count for value in random_position_hits),
        details=tuple(details),
    )
