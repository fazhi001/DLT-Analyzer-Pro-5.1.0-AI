from __future__ import annotations

from collections import Counter
from statistics import mean

from .models import Draw


def frequency(draws: list[Draw]) -> tuple[Counter[int], Counter[int]]:
    front = Counter(n for draw in draws for n in draw.front)
    back = Counter(n for draw in draws for n in draw.back)
    return front, back


def omission(draws: list[Draw]) -> tuple[dict[int, int], dict[int, int]]:
    front = {n: len(draws) for n in range(1, 36)}
    back = {n: len(draws) for n in range(1, 13)}
    for offset, draw in enumerate(reversed(draws)):
        for number in draw.front:
            if front[number] == len(draws):
                front[number] = offset
        for number in draw.back:
            if back[number] == len(draws):
                back[number] = offset
    return front, back


def draw_metrics(draw: Draw) -> dict[str, object]:
    odd = sum(number % 2 for number in draw.front)
    zones = (
        sum(1 <= n <= 12 for n in draw.front),
        sum(13 <= n <= 24 for n in draw.front),
        sum(25 <= n <= 35 for n in draw.front),
    )
    consecutive = sum(
        1
        for left, right in zip(draw.front, draw.front[1:])
        if right == left + 1
    )
    return {
        "sum": sum(draw.front),
        "odd": odd,
        "even": 5 - odd,
        "zones": zones,
        "consecutive": consecutive,
    }


def summary(draws: list[Draw]) -> dict[str, object]:
    if not draws:
        return {
            "count": 0,
            "average_sum": 0.0,
            "average_odd": 0.0,
            "consecutive_rate": 0.0,
        }
    metrics = [draw_metrics(draw) for draw in draws]
    return {
        "count": len(draws),
        "average_sum": mean(float(m["sum"]) for m in metrics),
        "average_odd": mean(float(m["odd"]) for m in metrics),
        "consecutive_rate": (
            sum(int(m["consecutive"]) > 0 for m in metrics) / len(metrics)
        ),
    }


def analysis_rows(draws: list[Draw]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    front_frequency, back_frequency = frequency(draws)
    front_omission, back_omission = omission(draws)

    front_rows = [
        {
            "number": number,
            "frequency": front_frequency[number],
            "omission": front_omission[number],
        }
        for number in range(1, 36)
    ]
    back_rows = [
        {
            "number": number,
            "frequency": back_frequency[number],
            "omission": back_omission[number],
        }
        for number in range(1, 13)
    ]
    return front_rows, back_rows
