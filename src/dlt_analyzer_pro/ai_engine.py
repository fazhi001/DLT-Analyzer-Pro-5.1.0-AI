from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ai_features import recent_five_years
from .ai_models import component_scores
from .ai_types import AIConfig, AIReport, ModelMetric
from .baseline_guard import blend_with_uniform
from .model_registry import ModelRegistry
from .stability import audit_training_pipeline, dataset_fingerprint, set_global_seed
from .models import Draw, Prediction


ProgressCallback = Callable[[str, float], None]


def _notify(callback: ProgressCallback | None, text: str, value: float) -> None:
    if callback is not None:
        callback(text, max(0.0, min(1.0, float(value))))


def _weighted_ensemble(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    first = next(iter(components.values()))
    output = np.zeros_like(first, dtype=float)
    for name, weight in weights.items():
        if name in components:
            output += float(weight) * components[name]
    output = np.maximum(output, 1e-6)
    output /= output.sum()
    return output


@dataclass(frozen=True, slots=True)
class HistoricalStructure:
    sum_mean: float
    sum_std: float
    odd_probabilities: dict[int, float]
    zone_probabilities: dict[tuple[int, int, int], float]
    consecutive_probability: float
    back_gap_mean: float
    back_gap_std: float


def learn_structure(draws: list[Draw]) -> HistoricalStructure:
    selected = recent_five_years(draws)
    front_sums = np.asarray([sum(draw.front) for draw in selected], dtype=float)
    odd_counts = [sum(number % 2 for number in draw.front) for draw in selected]
    zone_patterns = [
        (
            sum(1 <= n <= 12 for n in draw.front),
            sum(13 <= n <= 24 for n in draw.front),
            sum(25 <= n <= 35 for n in draw.front),
        )
        for draw in selected
    ]
    consecutive = [
        any(right == left + 1 for left, right in zip(draw.front, draw.front[1:]))
        for draw in selected
    ]
    back_gaps = np.asarray(
        [draw.back[1] - draw.back[0] for draw in selected],
        dtype=float,
    )
    odd_counter = Counter(odd_counts)
    zone_counter = Counter(zone_patterns)
    denominator = max(1, len(selected))
    return HistoricalStructure(
        sum_mean=float(front_sums.mean()),
        sum_std=max(8.0, float(front_sums.std())),
        odd_probabilities={
            key: (value + 1.0) / (denominator + 6.0)
            for key, value in odd_counter.items()
        },
        zone_probabilities={
            key: (value + 1.0) / (denominator + len(zone_counter) + 1.0)
            for key, value in zone_counter.items()
        },
        consecutive_probability=float(np.mean(consecutive)),
        back_gap_mean=float(back_gaps.mean()),
        back_gap_std=max(1.5, float(back_gaps.std())),
    )


def combination_fitness(
    front: tuple[int, ...],
    back: tuple[int, ...],
    front_probability: np.ndarray,
    back_probability: np.ndarray,
    structure: HistoricalStructure,
) -> float:
    front_index = np.asarray(front, dtype=int) - 1
    back_index = np.asarray(back, dtype=int) - 1
    base = float(np.log(front_probability[front_index] + 1e-12).sum())
    base += float(np.log(back_probability[back_index] + 1e-12).sum())

    total = sum(front)
    z_sum = (total - structure.sum_mean) / structure.sum_std
    odd = sum(number % 2 for number in front)
    zones = (
        sum(1 <= n <= 12 for n in front),
        sum(13 <= n <= 24 for n in front),
        sum(25 <= n <= 35 for n in front),
    )
    has_consecutive = any(
        right == left + 1 for left, right in zip(front, front[1:])
    )
    back_gap = back[1] - back[0]
    z_gap = (back_gap - structure.back_gap_mean) / structure.back_gap_std

    structure_score = -0.30 * z_sum * z_sum
    structure_score += math.log(structure.odd_probabilities.get(odd, 1e-4))
    structure_score += math.log(structure.zone_probabilities.get(zones, 1e-4))
    consecutive_probability = (
        structure.consecutive_probability
        if has_consecutive
        else 1.0 - structure.consecutive_probability
    )
    structure_score += 0.30 * math.log(max(1e-4, consecutive_probability))
    structure_score += -0.10 * z_gap * z_gap
    return base + 0.22 * structure_score


class MonteCarloSimulator:
    def __init__(
        self,
        front_probability: np.ndarray,
        back_probability: np.ndarray,
        structure: HistoricalStructure,
        seed: int,
    ) -> None:
        self.front_probability = np.asarray(front_probability, dtype=float)
        self.back_probability = np.asarray(back_probability, dtype=float)
        self.structure = structure
        self.rng = np.random.default_rng(seed)

    def _sample_without_replacement(
        self,
        probabilities: np.ndarray,
        sample_count: int,
        picks: int,
    ) -> np.ndarray:
        safe = np.maximum(probabilities, 1e-9)
        uniforms = np.maximum(
            self.rng.random((sample_count, len(safe))),
            1e-12,
        )
        keys = -np.log(uniforms) / safe
        selected = np.argpartition(keys, picks - 1, axis=1)[:, :picks]
        return np.sort(selected + 1, axis=1)

    def simulate(
        self,
        simulations: int,
        progress: ProgressCallback | None = None,
        batch_size: int = 25_000,
        keep_per_batch: int = 500,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...], float]]:
        simulations = max(1_000_000, int(simulations))
        best: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        completed = 0

        while completed < simulations:
            current = min(batch_size, simulations - completed)
            front_samples = self._sample_without_replacement(
                self.front_probability, current, 5
            )
            back_samples = self._sample_without_replacement(
                self.back_probability, current, 2
            )

            front_log = np.log(
                self.front_probability[front_samples - 1] + 1e-12
            ).sum(axis=1)
            back_log = np.log(
                self.back_probability[back_samples - 1] + 1e-12
            ).sum(axis=1)

            front_sum = front_samples.sum(axis=1)
            z_sum = (
                front_sum - self.structure.sum_mean
            ) / self.structure.sum_std
            odd_count = (front_samples % 2).sum(axis=1)
            back_gap = back_samples[:, 1] - back_samples[:, 0]
            z_gap = (
                back_gap - self.structure.back_gap_mean
            ) / self.structure.back_gap_std

            odd_score = np.asarray(
                [
                    math.log(
                        self.structure.odd_probabilities.get(int(value), 1e-4)
                    )
                    for value in odd_count
                ],
                dtype=float,
            )
            zone_patterns = np.column_stack(
                (
                    ((front_samples >= 1) & (front_samples <= 12)).sum(axis=1),
                    ((front_samples >= 13) & (front_samples <= 24)).sum(axis=1),
                    ((front_samples >= 25) & (front_samples <= 35)).sum(axis=1),
                )
            )
            zone_score = np.asarray(
                [
                    math.log(
                        self.structure.zone_probabilities.get(
                            tuple(int(v) for v in row),
                            1e-4,
                        )
                    )
                    for row in zone_patterns
                ],
                dtype=float,
            )
            score = front_log + back_log
            score += 0.22 * (
                -0.30 * z_sum * z_sum
                + odd_score
                + zone_score
                - 0.10 * z_gap * z_gap
            )

            keep = min(keep_per_batch, current)
            indices = np.argpartition(score, -keep)[-keep:]
            for index in indices:
                front = tuple(int(value) for value in front_samples[index])
                back = tuple(int(value) for value in back_samples[index])
                key = (front, back)
                value = float(score[index])
                if key not in best or value > best[key]:
                    best[key] = value

            completed += current
            _notify(
                progress,
                f"蒙特卡洛模拟 {completed:,}/{simulations:,}",
                completed / simulations,
            )

        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        return [
            (front, back, score)
            for (front, back), score in ranked
        ]


class GeneticOptimizer:
    def __init__(
        self,
        front_probability: np.ndarray,
        back_probability: np.ndarray,
        structure: HistoricalStructure,
        population_size: int,
        generations: int,
        seed: int,
    ) -> None:
        self.front_probability = np.asarray(front_probability, dtype=float)
        self.back_probability = np.asarray(back_probability, dtype=float)
        self.structure = structure
        self.population_size = max(80, int(population_size))
        self.generations = max(20, int(generations))
        self.rng = np.random.default_rng(seed)

    def fitness(self, chromosome) -> float:
        return combination_fitness(
            chromosome[0],
            chromosome[1],
            self.front_probability,
            self.back_probability,
            self.structure,
        )

    def _weighted_choice(
        self,
        values: list[int],
        probabilities: np.ndarray,
        count: int,
    ) -> tuple[int, ...]:
        available = list(values)
        chosen: list[int] = []
        for _ in range(count):
            weights = np.asarray(
                [probabilities[value - 1] for value in available],
                dtype=float,
            )
            weights /= weights.sum()
            index = int(self.rng.choice(len(available), p=weights))
            chosen.append(available.pop(index))
        return tuple(sorted(chosen))

    def _random_chromosome(self):
        front = self._weighted_choice(
            list(range(1, 36)), self.front_probability, 5
        )
        back = self._weighted_choice(
            list(range(1, 13)), self.back_probability, 2
        )
        return front, back

    def _crossover(self, parent_a, parent_b):
        front_pool = sorted(set(parent_a[0]) | set(parent_b[0]))
        back_pool = sorted(set(parent_a[1]) | set(parent_b[1]))
        if len(front_pool) < 5:
            front_pool = list(range(1, 36))
        if len(back_pool) < 2:
            back_pool = list(range(1, 13))
        front = self._weighted_choice(front_pool, self.front_probability, 5)
        back = self._weighted_choice(back_pool, self.back_probability, 2)
        return front, back

    def _mutate(self, chromosome, probability: float = 0.24):
        front = list(chromosome[0])
        back = list(chromosome[1])
        if self.rng.random() < probability:
            position = int(self.rng.integers(0, 5))
            available = [n for n in range(1, 36) if n not in front]
            weights = np.asarray(
                [self.front_probability[n - 1] for n in available],
                dtype=float,
            )
            weights /= weights.sum()
            front[position] = int(self.rng.choice(available, p=weights))
        if self.rng.random() < probability:
            position = int(self.rng.integers(0, 2))
            available = [n for n in range(1, 13) if n not in back]
            weights = np.asarray(
                [self.back_probability[n - 1] for n in available],
                dtype=float,
            )
            weights /= weights.sum()
            back[position] = int(self.rng.choice(available, p=weights))
        return tuple(sorted(front)), tuple(sorted(back))

    def optimize(
        self,
        seeds: list[tuple[tuple[int, ...], tuple[int, ...], float]],
        progress: ProgressCallback | None = None,
    ) -> list[tuple[tuple[int, ...], tuple[int, ...], float]]:
        population = [(front, back) for front, back, _ in seeds[: self.population_size]]
        while len(population) < self.population_size:
            population.append(self._random_chromosome())

        for generation in range(self.generations):
            ranked = sorted(
                ((chromosome, self.fitness(chromosome)) for chromosome in population),
                key=lambda item: item[1],
                reverse=True,
            )
            elite_count = max(12, self.population_size // 10)
            next_population = [chromosome for chromosome, _ in ranked[:elite_count]]
            tournament_pool = [chromosome for chromosome, _ in ranked[: max(40, self.population_size // 2)]]

            while len(next_population) < self.population_size:
                candidates_a = self.rng.choice(
                    len(tournament_pool), size=4, replace=False
                )
                candidates_b = self.rng.choice(
                    len(tournament_pool), size=4, replace=False
                )
                parent_a = max(
                    (tournament_pool[int(index)] for index in candidates_a),
                    key=self.fitness,
                )
                parent_b = max(
                    (tournament_pool[int(index)] for index in candidates_b),
                    key=self.fitness,
                )
                child = self._mutate(self._crossover(parent_a, parent_b))
                next_population.append(child)
            population = next_population
            _notify(
                progress,
                f"遗传算法第 {generation + 1}/{self.generations} 代",
                (generation + 1) / self.generations,
            )

        unique: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
        for chromosome in population:
            unique[chromosome] = max(
                unique.get(chromosome, -1e18),
                self.fitness(chromosome),
            )
        return [
            (front, back, score)
            for (front, back), score in sorted(
                unique.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]


def select_diverse(
    ranked: list[tuple[tuple[int, ...], tuple[int, ...], float]],
    count: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], float]]:
    selected: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
    remaining = list(ranked)
    while remaining and len(selected) < count:
        best_index = 0
        best_adjusted = -1e18
        for index, candidate in enumerate(remaining[:3000]):
            front, back, raw_score = candidate
            overlap_penalty = 0.0
            for chosen_front, chosen_back, _ in selected:
                overlap_penalty = max(
                    overlap_penalty,
                    0.42 * len(set(front) & set(chosen_front))
                    + 0.55 * len(set(back) & set(chosen_back)),
                )
            adjusted = raw_score - overlap_penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


class AIPredictionSystem:
    def __init__(self, config: AIConfig | None = None):
        self.config = config or AIConfig()

    def score_numbers(
        self,
        draws: list[Draw],
        include_ml: bool = True,
        fast_ml: bool = False,
        progress: ProgressCallback | None = None,
    ):
        selected = recent_five_years(draws)
        front_weights = self.config.normalized_zone_weights("front")
        back_weights = self.config.normalized_zone_weights("back")
        registry = ModelRegistry() if self.config.model_cache and include_ml and not fast_ml else None
        _notify(progress, "训练前区统计与机器学习模型", 0.05)
        front_components, front_metrics = component_scores(
            selected,
            "front",
            estimators=self.config.ml_estimators,
            random_state=self.config.seed,
            include_ml=include_ml,
            fast_ml=fast_ml,
            calibrate=self.config.probability_calibration,
            registry=registry,
            use_registry=bool(registry),
            persist_registry=bool(registry),
        )
        _notify(progress, "训练后区统计与机器学习模型", 0.15)
        back_components, back_metrics = component_scores(
            selected,
            "back",
            estimators=self.config.ml_estimators,
            random_state=self.config.seed + 1,
            include_ml=include_ml,
            fast_ml=fast_ml,
            calibrate=self.config.probability_calibration,
            registry=registry,
            use_registry=bool(registry),
            persist_registry=bool(registry),
        )
        front_probability = _weighted_ensemble(front_components, front_weights)
        back_probability = _weighted_ensemble(back_components, back_weights)
        front_probability = blend_with_uniform(
            front_probability, self.config.zone_model_share("front")
        )
        back_probability = blend_with_uniform(
            back_probability, self.config.zone_model_share("back")
        )
        return (
            front_probability,
            back_probability,
            tuple(front_metrics + back_metrics),
            front_components,
            back_components,
        )

    def predict(
        self,
        draws: list[Draw],
        progress: ProgressCallback | None = None,
    ) -> tuple[list[Prediction], AIReport]:
        start = time.perf_counter()
        if self.config.deterministic:
            set_global_seed(self.config.seed)
        audit = audit_training_pipeline(draws)
        if self.config.leakage_guard and not audit.passed:
            messages = "；".join(item.message for item in audit.issues if item.severity == "critical")
            raise RuntimeError(f"稳定性检查未通过，已阻止训练：{messages}")
        selected = recent_five_years(draws)
        (
            front_probability,
            back_probability,
            metrics,
            _,
            _,
        ) = self.score_numbers(selected, include_ml=True, progress=progress)
        structure = learn_structure(selected)

        simulator = MonteCarloSimulator(
            front_probability,
            back_probability,
            structure,
            seed=self.config.seed,
        )
        monte_carlo = simulator.simulate(
            self.config.simulations,
            progress=lambda text, value: _notify(
                progress, text, 0.20 + 0.58 * value
            ),
        )
        optimizer = GeneticOptimizer(
            front_probability,
            back_probability,
            structure,
            population_size=self.config.ga_population,
            generations=self.config.ga_generations,
            seed=self.config.seed + 17,
        )
        genetic = optimizer.optimize(
            monte_carlo,
            progress=lambda text, value: _notify(
                progress, text, 0.80 + 0.18 * value
            ),
        )
        combined = genetic + monte_carlo
        deduplicated: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            float,
        ] = {}
        for front, back, score in combined:
            key = (front, back)
            deduplicated[key] = max(deduplicated.get(key, -1e18), score)
        ranked = [
            (front, back, score)
            for (front, back), score in sorted(
                deduplicated.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        selected_combinations = select_diverse(
            ranked, self.config.prediction_count
        )
        scores = np.asarray([item[2] for item in selected_combinations], dtype=float)
        if scores.size and scores.max() > scores.min():
            display_scores = 70.0 + 30.0 * (
                scores - scores.min()
            ) / (scores.max() - scores.min())
        else:
            display_scores = np.full(len(selected_combinations), 85.0)

        predictions = [
            Prediction(
                front=front,
                back=back,
                score=round(float(display_scores[index]), 4),
                strategy=(
                    "AI集成模型（基线保护）"
                    if self.config.zone_model_share("front") < 0.999
                    or self.config.zone_model_share("back") < 0.999
                    else "AI集成模型"
                ),
            )
            for index, (front, back, _) in enumerate(selected_combinations)
        ]
        _notify(progress, "AI预测完成", 1.0)
        report = AIReport(
            dataset_count=len(selected),
            simulations=max(1_000_000, self.config.simulations),
            elapsed_seconds=time.perf_counter() - start,
            component_weights=self.config.normalized_weights(),
            model_metrics=metrics,
            front_scores={
                index + 1: float(value)
                for index, value in enumerate(front_probability)
            },
            back_scores={
                index + 1: float(value)
                for index, value in enumerate(back_probability)
            },
            weight_source="manual",
            dynamic_result=None,
            deterministic_seed=self.config.seed if self.config.deterministic else 0,
            dataset_fingerprint=audit.fingerprint,
            leakage_audit_passed=audit.passed,
            stability_notes=tuple(item.message for item in audit.issues),
            front_component_weights=self.config.normalized_zone_weights("front"),
            back_component_weights=self.config.normalized_zone_weights("back"),
            front_model_share=self.config.zone_model_share("front"),
            back_model_share=self.config.zone_model_share("back"),
            baseline_guard_notes=(
                f"前区模型概率占比 {self.config.zone_model_share('front'):.1%}",
                f"后区模型概率占比 {self.config.zone_model_share('back'):.1%}",
            ),
        )
        return predictions, report
