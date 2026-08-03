from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.metrics import log_loss

from .models import DigitDraw, DigitPrediction
from .paths import model_dir
from .holdout_validation import evaluate_final_holdout


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
FEATURE_VERSION = "digit-position-v2-walkforward"
_MAX_ML_WEIGHT = 0.70
_MIN_ABSOLUTE_IMPROVEMENT = 0.002
_MIN_RELATIVE_IMPROVEMENT = 0.0025


def prediction_brief(predictions: Iterable[DigitPrediction], limit: int = 3) -> str:
    """Return a compact, user-facing summary without implying a win guarantee."""
    rows = list(predictions)
    if not rows:
        return "尚未生成候选号码。"
    shown = "、".join(item.number_text for item in rows[:max(1, limit)])
    best = max(item.score for item in rows)
    return (
        f"已生成 {len(rows)} 注候选｜前 {min(len(rows), max(1, limit))} 注：{shown}"
        f"｜最高相对评分 {best:.2f}。结果仅供统计排序参考。"
    )


@dataclass(frozen=True, slots=True)
class PositionModelStatus:
    position: int
    position_name: str
    enabled: bool
    backend: str
    validation_logloss: float | None
    baseline_logloss: float | None
    reason: str
    ml_weight: float = 0.0
    validation_periods: int = 0
    fold_win_rate: float = 0.0


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
    ml_weight: float
    validation_periods: int
    fold_win_rate: float
    feature_version: str = FEATURE_VERSION


@dataclass(frozen=True, slots=True)
class _WalkForwardResult:
    baseline_probability: np.ndarray
    model_probability: np.ndarray
    targets: np.ndarray
    fold_slices: tuple[slice, ...]
    backend: str


@dataclass(frozen=True, slots=True)
class _BlendResult:
    enabled: bool
    ml_weight: float
    baseline_loss: float
    blended_loss: float
    model_loss: float
    fold_win_rate: float
    reason: str


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
    next_values = [
        int(values[index + 1])
        for index in range(len(values) - 1)
        if int(values[index]) == last
    ]
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

    recent_draws = list(draws[-5:])
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


def _fold_boundaries(sample_count: int, minimum_train: int = 30) -> list[tuple[int, int]]:
    validation_count = min(120, max(40, int(sample_count * 0.25)))
    validation_start = max(minimum_train, sample_count - validation_count)
    remaining = sample_count - validation_start
    if remaining < 20:
        return []
    fold_count = min(5, max(2, remaining // 20))
    boundaries = np.linspace(validation_start, sample_count, fold_count + 1, dtype=int)
    folds: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start and start >= minimum_train:
            folds.append((int(start), int(end)))
    return folds


def _walk_forward_validate(
    draws: list[DigitDraw],
    position: int,
    x: np.ndarray,
    y: np.ndarray,
    indices: list[int],
    seed: int,
) -> _WalkForwardResult | None:
    folds = _fold_boundaries(len(y))
    if not folds:
        return None

    baseline_parts: list[np.ndarray] = []
    model_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    output_slices: list[slice] = []
    backend = ""
    output_start = 0

    for fold_number, (start, end) in enumerate(folds):
        estimator, backend = _build_estimator(seed + fold_number)
        estimator.fit(x[:start], y[:start])
        model_probability = _map_probability(estimator, x[start:end])
        baseline_probability = np.vstack(
            [
                statistical_position_probability(draws[: indices[row]], position, "均衡模式")
                for row in range(start, end)
            ]
        )
        baseline_parts.append(baseline_probability)
        model_parts.append(model_probability)
        target_parts.append(y[start:end])
        output_end = output_start + (end - start)
        output_slices.append(slice(output_start, output_end))
        output_start = output_end

    if not target_parts:
        return None
    return _WalkForwardResult(
        baseline_probability=np.vstack(baseline_parts),
        model_probability=np.vstack(model_parts),
        targets=np.concatenate(target_parts),
        fold_slices=tuple(output_slices),
        backend=backend,
    )


def _blend_probability(
    baseline_probability: np.ndarray,
    model_probability: np.ndarray,
    ml_weight: float,
) -> np.ndarray:
    weight = min(_MAX_ML_WEIGHT, max(0.0, float(ml_weight)))
    blended = (1.0 - weight) * baseline_probability + weight * model_probability
    blended = np.clip(blended, 1e-12, None)
    blended /= blended.sum(axis=1, keepdims=True)
    return blended


def _optimize_blend(result: _WalkForwardResult) -> _BlendResult:
    """Select a blend on earlier folds, then judge it on a final untouched holdout."""
    total = len(result.targets)
    holdout_count = max(30, total // 4)
    selection_end = total - holdout_count
    if selection_end < 20:
        baseline_loss = float(
            log_loss(result.targets, result.baseline_probability, labels=list(range(10)))
        )
        return _BlendResult(
            enabled=False,
            ml_weight=0.0,
            baseline_loss=baseline_loss,
            blended_loss=baseline_loss,
            model_loss=baseline_loss,
            fold_win_rate=0.0,
            reason="独立留出期不足，自动停用AI",
        )

    selection_targets = result.targets[:selection_end]
    selection_baseline = result.baseline_probability[:selection_end]
    selection_model = result.model_probability[:selection_end]
    baseline_loss = float(
        log_loss(selection_targets, selection_baseline, labels=list(range(10)))
    )
    model_loss = float(
        log_loss(selection_targets, selection_model, labels=list(range(10)))
    )

    best_weight = 0.0
    best_loss = baseline_loss
    for weight in np.linspace(0.05, _MAX_ML_WEIGHT, 14):
        probability = _blend_probability(selection_baseline, selection_model, float(weight))
        loss = float(log_loss(selection_targets, probability, labels=list(range(10))))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)

    selected_probability = _blend_probability(
        result.baseline_probability[selection_end:],
        result.model_probability[selection_end:],
        best_weight,
    )
    holdout_baseline = -np.log(
        np.clip(
            result.baseline_probability[selection_end:][
                np.arange(holdout_count), result.targets[selection_end:]
            ],
            1e-12,
            1.0,
        )
    )
    holdout_candidate = -np.log(
        np.clip(
            selected_probability[
                np.arange(holdout_count), result.targets[selection_end:]
            ],
            1e-12,
            1.0,
        )
    )
    holdout = evaluate_final_holdout(
        holdout_candidate,
        holdout_baseline,
        minimum_holdout=holdout_count,
    )
    enabled = bool(best_weight > 0.0 and holdout.enabled)
    return _BlendResult(
        enabled=enabled,
        ml_weight=best_weight if enabled else 0.0,
        baseline_loss=holdout.baseline_loss,
        blended_loss=holdout.candidate_loss if enabled else holdout.baseline_loss,
        model_loss=model_loss,
        fold_win_rate=1.0 if holdout.enabled else 0.0,
        reason=holdout.reason,
    )

def _enumerated_digits(position_count: int) -> np.ndarray:
    total = 10 ** position_count
    values = np.arange(total, dtype=np.int32)
    digits = np.empty((total, position_count), dtype=np.int8)
    for position in range(position_count - 1, -1, -1):
        digits[:, position] = values % 10
        values //= 10
    return digits


def enumerate_digit_candidates(
    probabilities: list[np.ndarray],
    historical_sums: np.ndarray,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    position_count = len(probabilities)
    digits = _enumerated_digits(position_count)
    log_scores = np.zeros(len(digits), dtype=float)
    for position, probability in enumerate(probabilities):
        log_scores += np.log(np.clip(probability[digits[:, position]], 1e-12, None))

    sum_mean = (
        float(historical_sums.mean())
        if len(historical_sums)
        else 4.5 * position_count
    )
    sum_std = float(historical_sums.std()) if len(historical_sums) else 3.0
    sum_std = max(1.0, sum_std)
    z_score = np.abs(digits.sum(axis=1) - sum_mean) / sum_std
    log_scores -= 0.08 * z_score

    sorted_digits = np.sort(digits, axis=1)
    unique_count = 1 + np.sum(np.diff(sorted_digits, axis=1) != 0, axis=1)
    if strategy == "稳定模式":
        log_scores += 0.05 * (unique_count >= max(2, position_count - 1))
    elif strategy == "偏冷模式":
        log_scores += 0.03 * (unique_count < position_count)

    order = np.argsort(-log_scores, kind="stable")
    return digits[order], log_scores[order]


class DigitPredictionEngine:
    def __init__(self, game: str, seed: int = 20260724, model_base_dir: Path | None = None):
        game = game.lower()
        if game not in GAME_POSITIONS:
            raise ValueError(f"不支持的玩法：{game}")
        self.game = game
        self.position_count = GAME_POSITIONS[game]
        self.seed = int(seed)
        self.model_base_dir = model_base_dir

    def train_models(self, draws: list[DigitDraw], force: bool = False) -> DigitModelReport:
        valid = [draw for draw in draws if draw.game == self.game]
        if len(valid) < 100:
            statuses = tuple(
                PositionModelStatus(
                    position=index,
                    position_name=POSITION_NAMES[self.game][index],
                    enabled=False,
                    backend="统计融合",
                    validation_logloss=None,
                    baseline_logloss=None,
                    reason="至少需要100期数据进行滚动样本外验证",
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
                                enabled=bundle.ml_weight > 0,
                                backend=bundle.backend,
                                validation_logloss=bundle.validation_logloss,
                                baseline_logloss=bundle.baseline_logloss,
                                reason="已加载与当前数据一致的滚动验证模型",
                                ml_weight=bundle.ml_weight,
                                validation_periods=bundle.validation_periods,
                                fold_win_rate=bundle.fold_win_rate,
                            )
                        )
                        continue
                except Exception:
                    pass

            x, y, indices = _build_dataset(valid, position)
            if len(y) < 65 or len(np.unique(y)) < 5:
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

            validation = _walk_forward_validate(
                valid,
                position,
                x,
                y,
                indices,
                self.seed + position * 100,
            )
            if validation is None:
                statuses.append(
                    PositionModelStatus(
                        position=position,
                        position_name=POSITION_NAMES[self.game][position],
                        enabled=False,
                        backend="统计融合",
                        validation_logloss=None,
                        baseline_logloss=None,
                        reason="无法形成足够的滚动验证时间折",
                    )
                )
                continue

            blend = _optimize_blend(validation)
            if blend.enabled:
                estimator, backend = _build_estimator(self.seed + position)
                estimator.fit(x, y)
                bundle = _ModelBundle(
                    model=estimator,
                    classes=np.asarray(estimator.classes_, dtype=int),
                    validation_logloss=blend.blended_loss,
                    baseline_logloss=blend.baseline_loss,
                    backend=backend,
                    fingerprint=fingerprint,
                    ml_weight=blend.ml_weight,
                    validation_periods=len(validation.targets),
                    fold_win_rate=blend.fold_win_rate,
                )
                joblib.dump(bundle, path)
            else:
                backend = "统计融合"
                if path.exists():
                    path.unlink(missing_ok=True)

            statuses.append(
                PositionModelStatus(
                    position=position,
                    position_name=POSITION_NAMES[self.game][position],
                    enabled=blend.enabled,
                    backend=backend,
                    validation_logloss=blend.blended_loss,
                    baseline_logloss=blend.baseline_loss,
                    reason=blend.reason,
                    ml_weight=blend.ml_weight,
                    validation_periods=len(validation.targets),
                    fold_win_rate=blend.fold_win_rate,
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
                        and bundle.ml_weight > 0
                    ):
                        x = _feature_vector(valid, position).reshape(1, -1)
                        ml_probability = _map_probability(bundle.model, x)[0]
                        probability = _normalize(
                            (1.0 - bundle.ml_weight) * statistical
                            + bundle.ml_weight * ml_probability
                        )
                        enabled_count += 1
                        status = PositionModelStatus(
                            position=position,
                            position_name=POSITION_NAMES[self.game][position],
                            enabled=True,
                            backend=bundle.backend,
                            validation_logloss=bundle.validation_logloss,
                            baseline_logloss=bundle.baseline_logloss,
                            reason=f"滚动验证通过，AI动态权重{bundle.ml_weight:.0%}",
                            ml_weight=bundle.ml_weight,
                            validation_periods=bundle.validation_periods,
                            fold_win_rate=bundle.fold_win_rate,
                        )
                except Exception:
                    pass
            probabilities.append(probability)
            statuses.append(status)

        mode = (
            f"可信动态融合（{enabled_count}/{self.position_count}位）"
            if enabled_count
            else "统计融合"
        )
        return probabilities, mode, tuple(statuses)

    def generate(
        self,
        draws: list[DigitDraw],
        count: int = 10,
        strategy: str = "均衡模式",
        candidate_count: int = 5000,
        use_ml: bool = True,
    ) -> list[DigitPrediction]:
        # candidate_count is retained for API compatibility. V2 always evaluates
        # the complete PL3/PL5 outcome space, so no candidate can be missed.
        del candidate_count
        count = max(1, min(int(count), 200))
        probabilities, mode, _ = self.position_probabilities(draws, strategy, use_ml=use_ml)
        historical_sums = np.asarray([sum(draw.digits) for draw in draws[-300:]], dtype=float)
        ranked_digits, ranked_scores = enumerate_digit_candidates(
            probabilities,
            historical_sums,
            strategy,
        )
        if not len(ranked_scores):
            return []

        top = float(ranked_scores[0])
        scaled = np.exp(np.clip(ranked_scores - top, -50, 0))
        scaled /= scaled.max() if scaled.max() > 0 else 1.0

        selected: list[DigitPrediction] = []
        selected_digits: list[tuple[int, ...]] = []
        for index, row in enumerate(ranked_digits):
            digits = tuple(int(value) for value in row)
            if any(
                sum(a == b for a, b in zip(digits, existing)) >= self.position_count - 1
                for existing in selected_digits
            ):
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
            selected_digits.append(digits)
            if len(selected) >= count:
                break

        if len(selected) < count:
            existing = set(selected_digits)
            for index, row in enumerate(ranked_digits):
                digits = tuple(int(value) for value in row)
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
                existing.add(digits)
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
