from __future__ import annotations

import random
from dataclasses import dataclass

from .analytics import frequency, omission
from .models import Draw, Prediction


STRATEGIES = {
    "均衡模式": (0.35, 0.30, 0.35),
    "近期热号": (0.20, 0.60, 0.20),
    "遗漏回补": (0.20, 0.20, 0.60),
    "冷热混合": (0.40, 0.35, 0.25),
}


@dataclass(frozen=True, slots=True)
class NumberScores:
    front: dict[int, float]
    back: dict[int, float]


def _normalize(values: dict[int, float]) -> dict[int, float]:
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 0.5 for key in values}
    return {
        key: (value - low) / (high - low)
        for key, value in values.items()
    }


class PredictionEngine:
    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def score_numbers(self, draws: list[Draw], strategy: str) -> NumberScores:
        if len(draws) < 30:
            raise ValueError("至少需要30期历史数据")
        if strategy not in STRATEGIES:
            raise ValueError(f"未知策略：{strategy}")

        all_front, all_back = frequency(draws)
        recent_draws = draws[-60:]
        recent_front, recent_back = frequency(recent_draws)
        omit_front, omit_back = omission(draws)

        all_front_n = _normalize({n: all_front[n] for n in range(1, 36)})
        recent_front_n = _normalize({n: recent_front[n] for n in range(1, 36)})
        omit_front_n = _normalize(omit_front)

        all_back_n = _normalize({n: all_back[n] for n in range(1, 13)})
        recent_back_n = _normalize({n: recent_back[n] for n in range(1, 13)})
        omit_back_n = _normalize(omit_back)

        long_weight, recent_weight, omit_weight = STRATEGIES[strategy]
        front_scores = {
            n: (
                long_weight * all_front_n[n]
                + recent_weight * recent_front_n[n]
                + omit_weight * omit_front_n[n]
            )
            for n in range(1, 36)
        }
        back_scores = {
            n: (
                long_weight * all_back_n[n]
                + recent_weight * recent_back_n[n]
                + omit_weight * omit_back_n[n]
            )
            for n in range(1, 13)
        }
        return NumberScores(front_scores, back_scores)

    def _weighted_sample(
        self,
        numbers: range,
        scores: dict[int, float],
        count: int,
    ) -> tuple[int, ...]:
        available = list(numbers)
        chosen: list[int] = []
        for _ in range(count):
            weights = [max(0.02, scores[n]) for n in available]
            number = self.random.choices(available, weights=weights, k=1)[0]
            chosen.append(number)
            available.remove(number)
        return tuple(sorted(chosen))

    @staticmethod
    def _structure_adjustment(front: tuple[int, ...], back: tuple[int, ...]) -> float:
        adjustment = 0.0
        total = sum(front)
        odd = sum(n % 2 for n in front)
        zones = (
            sum(1 <= n <= 12 for n in front),
            sum(13 <= n <= 24 for n in front),
            sum(25 <= n <= 35 for n in front),
        )
        if 70 <= total <= 120:
            adjustment += 0.15
        else:
            adjustment -= 0.20
        if odd in (2, 3):
            adjustment += 0.12
        else:
            adjustment -= 0.10
        if all(zone > 0 for zone in zones):
            adjustment += 0.10
        if max(zones) >= 4:
            adjustment -= 0.12
        if any(right == left + 1 for left, right in zip(front, front[1:])):
            adjustment += 0.03
        if back[1] - back[0] >= 3:
            adjustment += 0.03
        return adjustment

    def generate(
        self,
        draws: list[Draw],
        count: int = 10,
        strategy: str = "均衡模式",
        candidate_count: int = 4000,
    ) -> list[Prediction]:
        count = max(1, min(int(count), 50))
        scores = self.score_numbers(draws, strategy)
        candidates: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            float,
        ] = {}

        for _ in range(max(candidate_count, count * 100)):
            front = self._weighted_sample(range(1, 36), scores.front, 5)
            back = self._weighted_sample(range(1, 13), scores.back, 2)
            raw_score = (
                sum(scores.front[n] for n in front)
                + sum(scores.back[n] for n in back)
                + self._structure_adjustment(front, back)
            )
            key = (front, back)
            if key not in candidates or raw_score > candidates[key]:
                candidates[key] = raw_score

        ranked = sorted(
            candidates.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        selected: list[Prediction] = []
        for (front, back), score in ranked:
            if any(
                len(set(front) & set(existing.front)) >= 4
                and len(set(back) & set(existing.back)) >= 1
                for existing in selected
            ):
                continue
            selected.append(
                Prediction(
                    front=front,
                    back=back,
                    score=round(float(score), 4),
                    strategy=strategy,
                )
            )
            if len(selected) == count:
                break

        if len(selected) < count:
            existing_keys = {(p.front, p.back) for p in selected}
            for (front, back), score in ranked:
                if (front, back) in existing_keys:
                    continue
                selected.append(
                    Prediction(
                        front=front,
                        back=back,
                        score=round(float(score), 4),
                        strategy=strategy,
                    )
                )
                if len(selected) == count:
                    break
        return selected


def next_issue(latest: str | None) -> str:
    if not latest:
        return "下一期"
    try:
        return str(int(latest) + 1).zfill(len(latest))
    except ValueError:
        return "下一期"
