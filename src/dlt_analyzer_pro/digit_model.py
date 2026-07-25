from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.metrics import log_loss

from .models import DigitDraw, DigitPrediction
from .paths import model_dir


GAME_NAMES = {"pl3": "排列三", "pl5": "排列五"}
GAME_POSITIONS = {"pl3": 3, "pl5": 5}
POSITION_NAMES = {
    "pl3": ("百位", "十位", "个位"),
    "pl5": ("万位", "千位", "百位", "十位", "个位"),
}
STRATEGIES = {
    "均衡模式": (0.25, 0.30, 0.25, 0.10, 0.10),
    "稳定模式": (0.40, 0.25, 0.15, 0.10, 0.10),
    "偏热模式": (0.10, 0.30, 0.40, 0.10, 0.10),
    "偏冷模式": (0.18, 0.14, 0.08, 0.05, 0.55),
}
FEATURE_VERSION = "digit-position-v1"


@dataclass(frozen=True, slots=True)
class PositionModelStatus:
    position: int
    position_name: str
    enabled: bool
    backend: str
    validation_logloss: float | None
    baseline_logloss: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class DigitModelReport:
    game: str
    fingerprint: str
    statuses: tuple[PositionModelStatus, ...]

    @property
    def enabled_count(self) -> int:
        return sum(status.enabled for status in self.statuses)


@dataclass(slots=True)
class _ModelBundle:
    model: object
    classes: np.ndarray
    validation_logloss: float
    baseline_logloss: float
    backend: str
    fingerprint: str
    feature_version: str = FEATURE_VERSION


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 1e-12, None)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full(10, 0.1, dtype=float)
    return values / total


def dataset_fingerprint(draws: Iterable[DigitDraw], game: str) -> str:
    digest = hashlib.sha256()
    digest.update(game.encode("ascii"))
    digest.update(FEATURE_VERSION.encode("ascii"))
    for draw in draws:
        digest.update(draw.issue.encode("utf-8"))
        digest.update(bytes(draw.digits))
    return digest.hexdigest()[:20]


def _position_matrix(draws: list[DigitDraw], position: int) -> np.ndarray:
    return np.asarray([draw.digits[position] for draw in draws], dtype=int)


def _counts(values: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    result = np.bincount(values.astype(int), minlength=10).astype(float)
    result += float(alpha)
    return _normalize(result)


def _omission(values: np.ndarray) -> np.ndarray:
    gaps = np.zeros(10, dtype=float)
    for digit in range(10):
        indices = np.where(values == digit)[0]
        gaps[digit] = len(values) if len(indices) == 0 else len(values) - 1 - int(indices[-1])
    return _normalize(np.log1p(gaps) + 0.2)


def _transition(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.full(10, 0.1, dtype=float)
    last = int(values[-1])
    next_values = [int(values[index + 1]) for index in range(len(values) - 1) if int(values[index]) == last]
    if not next_values:
        return _counts(values[-30:], alpha=1.0)
    return _counts(np.asarray(next_values, dtype=int), alpha=1.0)


def statistical_position_probability(
    draws: list[DigitDraw],
    position: int,
    strategy: str = "均衡模式",
) -> np.ndarray:
    if not draws:
        return np.full(10, 0.1, dtype=float)
    values = _position_matrix(draws, position)
    long_p = _counts(values, alpha=1.5)
    recent30 = _counts(values[-30:], alpha=1.2)
    recent10 = _counts(values[-10:], alpha=1.0)
    transition = _transition(values)
    omission = _omission(values)
    weights = STRATEGIES.get(strategy, STRATEGIES["均衡模式"])
    probability = (
        weights[0] * long_p
        + weights[1] * recent30
        + weights[2] * recent10
        + weights[3] * transition
        + weights[4] * omission
    )
    return _normalize(probability)


def _omission_features(values: np.ndarray) -> np.ndarray:
    gaps = np.zeros(10, dtype=float)
    for digit in range(10):
        indices = np.where(values == digit)[0]
        gaps[digit] = len(values) if len(indices) == 0 else len(values) - 1 - int(indices[-1])
    scale = max(1.0, float(len(values)))
    return np.log1p(gaps) / math.log1p(scale)


def _feature_vector(draws: list[DigitDraw], position: int) -> np.ndarray:
    if not draws:
        raise ValueError("特征计算至少需要1期数据")
    position_count = len(draws[-1].digits)
    values = _position_matrix(draws, position)
    features: list[float] = []

    for lag in range(1, 7):
        features.append(float(values[-lag]) / 9.0 if len(values) >= lag else 0.5)

    latest = list(draws[-1].digits) + [0] * (5 - position_count)
    features.extend(float(value) / 9.0 for value in latest[:5])

    for window in (10, 30, 100):
        probability = _counts(values[-window:], alpha=1.0)
        features.extend(probability.tolist())

    features.extend(_omission_features(values).tolist())
    features.extend(_transition(values).tolist())

    recent_draws = draws[-5:]
    for draw in recent_draws:
        digits = np.asarray(draw.digits, dtype=float)
        features.extend(
            [
                float(digits.sum()) / (9.0 * len(digits)),
                float(digits.max() - digits.min()) / 9.0,
                float(np.mean(digits % 2)),
                float(len(set(draw.digits))) / len(draw.digits),
            ]
        )
    while len(recent_draws) < 5:
        features.extend([0.5, 0.5, 0.5, 0.5])
        recent_draws.append(draws[-1])

    return np.asarray(features, dtype=float)


def _build_dataset(draws: list[DigitDraw], position: int, min_history: int = 35):
    x_rows: list[np.ndarray] = []
    labels: list[int] = []
    indices: list[int] = []
    for index in range(min_history, len(draws)):
        x_rows.append(_feature_vector(draws[:index], position))
        labels.append(int(draws[index].digits[position]))
        indices.append(index)
    if not x_rows:
        return np.empty((0, 0)), np.empty((0,), dtype=int), []
    return np.vstack(x_rows), np.asarray(labels, dtype=int), indices


def _build_estimator(seed: int):
    try:
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(
                objective="multiclass",
                num_class=10,
                n_estimators=120,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=5,
                min_child_samples=12,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.2,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            ),
            "LightGBM",
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=140,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            ),
            "Scikit-learn HGB",
        )


def _map_probability(model, x: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    classes = np.asarray(model.classes_, dtype=int)
    result = np.full((raw.shape[0], 10), 1e-6, dtype=float)
    for source_index, label in enumerate(classes):
        if 0 <= int(label) <= 9:
            result[:, int(label)] = raw[:, source_index]
    result /= result.sum(axis=1, keepdims=True)
    return result


def _bundle_path(game: str, position: int, base_dir: Path | None = None) -> Path:
    directory = Path(base_dir or model_dir()) / game
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"position_{position + 1}.joblib"


class DigitPredictionEngine:
    def __init__(self, game: str, seed: int = 20260724, model_base_dir: Path | None = None):
        game = game.lower()
        if game not in GAME_POSITIONS:
            raise ValueError(f"不支持的玩法：{game}")
        self.game = game
        self.position_count = GAME_POSITIONS[game]
        self.random = random.Random(seed)
        self.seed = int(seed)
        self.model_base_dir = model_base_dir

    def train_models(self, draws: list[DigitDraw], force: bool = False) -> DigitModelReport:
        valid = [draw for draw in draws if draw.game == self.game]
        if len(valid) < 80:
            statuses = tuple(
                PositionModelStatus(
                    position=index,
                    position_name=POSITION_NAMES[self.game][index],
                    enabled=False,
                    backend="统计融合",
                    validation_logloss=None,
                    baseline_logloss=None,
                    reason="至少需要80期数据才能训练可信位置模型",
                )
                for index in range(self.position_count)
            )
            return DigitModelReport(self.game, dataset_fingerprint(valid, self.game), statuses)

        fingerprint = dataset_fingerprint(valid, self.game)
        statuses: list[PositionModelStatus] = []
        for position in range(self.position_count):
            path = _bundle_path(self.game, position, self.model_base_dir)
            if not force and path.exists():
                try:
                    bundle = joblib.load(path)
                    if (
                        isinstance(bundle, _ModelBundle)
                        and bundle.fingerprint == fingerprint
                        and bundle.feature_version == FEATURE_VERSION
                    ):
                        statuses.append(
                            PositionModelStatus(
                                position=position,
                                position_name=POSITION_NAMES[self.game][position],
                                enabled=True,
                                backend=bundle.backend,
                                validation_logloss=bundle.validation_logloss,
                                baseline_logloss=bundle.baseline_logloss,
                                reason="已加载与当前数据一致的模型",
                            )
                        )
                        continue
                except Exception:
                    pass

            x, y, indices = _build_dataset(valid, position)
            if len(y) < 45 or len(np.unique(y)) < 5:
                statuses.append(
                    PositionModelStatus(
                        position=position,
                        position_name=POSITION_NAMES[self.game][position],
                        enabled=False,
                        backend="统计融合",
                        validation_logloss=None,
                        baseline_logloss=None,
                        reason="有效训练样本或类别不足",
                    )
                )
                continue

            validation_count = min(80, max(20, int(len(y) * 0.2)))
            split = len(y) - validation_count
            if split < 30:
                split = max(20, len(y) - 20)
            estimator, backend = _build_estimator(self.seed + position)
            estimator.fit(x[:split], y[:split])
            model_probability = _map_probability(estimator, x[split:])
            model_loss = float(log_loss(y[split:], model_probability, labels=list(range(10))))

            baseline_probability = np.vstack(
                [
                    statistical_position_probability(valid[:indices[row]], position, "均衡模式")
                    for row in range(split, len(indices))
                ]
            )
            baseline_loss = float(log_loss(y[split:], baseline_probability, labels=list(range(10))))
            enabled = model_loss < baseline_loss * 0.995
            reason = (
                "样本外Log Loss优于统计基线"
                if enabled
                else "样本外表现未优于统计基线，自动停用AI"
            )
            if enabled:
                estimator.fit(x, y)
                bundle = _ModelBundle(
                    model=estimator,
                    classes=np.asarray(estimator.classes_, dtype=int),
                    validation_logloss=model_loss,
                    baseline_logloss=baseline_loss,
                    backend=backend,
                    fingerprint=fingerprint,
                )
                joblib.dump(bundle, path)
            elif path.exists():
                path.unlink(missing_ok=True)

            statuses.append(
                PositionModelStatus(
                    position=position,
                    position_name=POSITION_NAMES[self.game][position],
                    enabled=enabled,
                    backend=backend if enabled else "统计融合",
                    validation_logloss=model_loss,
                    baseline_logloss=baseline_loss,
                    reason=reason,
                )
            )
        return DigitModelReport(self.game, fingerprint, tuple(statuses))

    def position_probabilities(
        self,
        draws: list[DigitDraw],
        strategy: str = "均衡模式",
        use_ml: bool = True,
    ) -> tuple[list[np.ndarray], str, tuple[PositionModelStatus, ...]]:
        valid = [draw for draw in draws if draw.game == self.game]
        if not valid:
            raise ValueError(f"{GAME_NAMES[self.game]}暂无历史数据")
        fingerprint = dataset_fingerprint(valid, self.game)
        probabilities: list[np.ndarray] = []
        statuses: list[PositionModelStatus] = []
        enabled_count = 0

        for position in range(self.position_count):
            statistical = statistical_position_probability(valid, position, strategy)
            probability = statistical
            status = PositionModelStatus(
                position=position,
                position_name=POSITION_NAMES[self.game][position],
                enabled=False,
                backend="统计融合",
                validation_logloss=None,
                baseline_logloss=None,
                reason="使用统计融合",
            )
            if use_ml:
                path = _bundle_path(self.game, position, self.model_base_dir)
                try:
                    bundle = joblib.load(path)
                    if (
                        isinstance(bundle, _ModelBundle)
                        and bundle.fingerprint == fingerprint
                        and bundle.feature_version == FEATURE_VERSION
                    ):
                        x = _feature_vector(valid, position).reshape(1, -1)
                        ml_probability = _map_probability(bundle.model, x)[0]
                        probability = _normalize(0.65 * statistical + 0.35 * ml_probability)
                        enabled_count += 1
                        status = PositionModelStatus(
                            position=position,
                            position_name=POSITION_NAMES[self.game][position],
                            enabled=True,
                            backend=bundle.backend,
                            validation_logloss=bundle.validation_logloss,
                            baseline_logloss=bundle.baseline_logloss,
                            reason="AI通过样本外验证，与统计概率融合",
                        )
                except Exception:
                    pass
            probabilities.append(probability)
            statuses.append(status)

        mode = f"AI+统计融合（{enabled_count}/{self.position_count}位）" if enabled_count else "统计融合"
        return probabilities, mode, tuple(statuses)

    def generate(
        self,
        draws: list[DigitDraw],
        count: int = 10,
        strategy: str = "均衡模式",
        candidate_count: int = 5000,
        use_ml: bool = True,
    ) -> list[DigitPrediction]:
        count = max(1, min(int(count), 200))
        probabilities, mode, _ = self.position_probabilities(draws, strategy, use_ml=use_ml)
        historical_sums = np.asarray([sum(draw.digits) for draw in draws[-300:]], dtype=float)
        sum_mean = float(historical_sums.mean()) if len(historical_sums) else 4.5 * self.position_count
        sum_std = float(historical_sums.std()) if len(historical_sums) else 3.0
        sum_std = max(1.0, sum_std)

        candidates: dict[tuple[int, ...], float] = {}
        for _ in range(max(candidate_count, count * 120)):
            digits = tuple(
                self.random.choices(range(10), weights=probability.tolist(), k=1)[0]
                for probability in probabilities
            )
            log_score = sum(math.log(max(1e-12, probabilities[index][digit])) for index, digit in enumerate(digits))
            z_score = abs(sum(digits) - sum_mean) / sum_std
            structure = -0.08 * z_score
            unique_count = len(set(digits))
            if strategy == "稳定模式" and unique_count >= max(2, self.position_count - 1):
                structure += 0.05
            if strategy == "偏冷模式" and unique_count < self.position_count:
                structure += 0.03
            score = log_score + structure
            if digits not in candidates or score > candidates[digits]:
                candidates[digits] = score

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return []
        raw = np.asarray([score for _, score in ranked], dtype=float)
        top = float(raw.max())
        scaled = np.exp(np.clip(raw - top, -50, 0))
        scaled /= scaled.max() if scaled.max() > 0 else 1.0

        selected: list[DigitPrediction] = []
        for index, (digits, _) in enumerate(ranked):
            if any(sum(a == b for a, b in zip(digits, item.digits)) >= self.position_count - 1 for item in selected):
                continue
            selected.append(
                DigitPrediction(
                    game=self.game,
                    digits=digits,
                    score=round(float(scaled[index] * 100.0), 4),
                    strategy=strategy,
                    model_mode=mode,
                )
            )
            if len(selected) >= count:
                break
        if len(selected) < count:
            existing = {item.digits for item in selected}
            for index, (digits, _) in enumerate(ranked):
                if digits in existing:
                    continue
                selected.append(
                    DigitPrediction(
                        game=self.game,
                        digits=digits,
                        score=round(float(scaled[index] * 100.0), 4),
                        strategy=strategy,
                        model_mode=mode,
                    )
                )
                if len(selected) >= count:
                    break
        return selected


def digit_analysis_rows(
    draws: list[DigitDraw],
    strategy: str = "均衡模式",
) -> list[dict[str, object]]:
    if not draws:
        return []
    game = draws[-1].game
    engine = DigitPredictionEngine(game)
    probabilities, mode, _ = engine.position_probabilities(draws, strategy, use_ml=True)
    rows: list[dict[str, object]] = []
    for position in range(GAME_POSITIONS[game]):
        values = _position_matrix(draws, position)
        counts = np.bincount(values, minlength=10)
        omissions = _omission_features(values)
        for digit in range(10):
            last_indices = np.where(values == digit)[0]
            gap = len(values) if len(last_indices) == 0 else len(values) - 1 - int(last_indices[-1])
            rows.append(
                {
                    "position": POSITION_NAMES[game][position],
                    "digit": digit,
                    "count": int(counts[digit]),
                    "frequency": float(counts[digit] / len(values)),
                    "omission": int(gap),
                    "probability": float(probabilities[position][digit]),
                    "mode": mode,
                }
            )
    return rows
