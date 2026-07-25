from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .ai_features import build_feature_dataset, recent_five_years
from .models import Draw


@dataclass(frozen=True, slots=True)
class StabilityIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class StabilityAuditResult:
    passed: bool
    fingerprint: str
    draw_count: int
    checked_at: str
    issues: tuple[StabilityIssue, ...]

    @property
    def critical_count(self) -> int:
        return sum(item.severity == "critical" for item in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "warning" for item in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "fingerprint": self.fingerprint,
            "draw_count": self.draw_count,
            "checked_at": self.checked_at,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "issues": [asdict(item) for item in self.issues],
        }


def set_global_seed(seed: int) -> int:
    """Set deterministic seeds for Python and NumPy.

    XGBoost and LightGBM receive the same seed through their estimator
    parameters. Thread counts are intentionally kept small in model code to
    reduce non-determinism across machines.
    """
    value = int(seed)
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    np.random.seed(value)
    return value


def dataset_fingerprint(draws: list[Draw]) -> str:
    digest = hashlib.sha256()
    for draw in draws:
        payload = (
            draw.issue,
            draw.draw_date.isoformat() if draw.draw_date else "",
            *draw.front,
            *draw.back,
        )
        digest.update("|".join(map(str, payload)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def training_fingerprint(
    draws: list[Draw],
    zone: str,
    model_name: str,
    estimators: int,
    seed: int,
    calibration: bool = True,
) -> str:
    selected = recent_five_years(draws)
    base = dataset_fingerprint(selected)
    payload = f"{base}|{zone}|{model_name}|{int(estimators)}|{int(seed)}|cal={int(bool(calibration))}|v4.1"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prefix_invariance_check(draws: list[Draw], zone: str) -> StabilityIssue | None:
    selected = recent_five_years(draws)
    if len(selected) < 120:
        return StabilityIssue(
            "warning",
            f"{zone}_insufficient_history",
            "历史样本不足，未执行完整前缀不变性检查。",
        )

    full = build_feature_dataset(selected, zone)
    prefix = build_feature_dataset(selected[:-1], zone)
    overlap = len(prefix.y)
    if overlap <= 0:
        return StabilityIssue(
            "critical",
            f"{zone}_empty_overlap",
            "特征数据没有可验证的时间重叠区间。",
        )
    if not np.array_equal(full.y[:overlap], prefix.y):
        return StabilityIssue(
            "critical",
            f"{zone}_label_leakage",
            "加入未来一期后，既有标签发生变化，存在标签泄漏风险。",
        )
    if not np.allclose(full.X[:overlap], prefix.X, atol=1e-7, rtol=1e-7):
        return StabilityIssue(
            "critical",
            f"{zone}_feature_leakage",
            "加入未来一期后，历史特征发生变化，存在未来信息泄漏风险。",
        )
    return None


def audit_training_pipeline(draws: list[Draw]) -> StabilityAuditResult:
    issues: list[StabilityIssue] = []
    if len(draws) < 100:
        issues.append(
            StabilityIssue("critical", "insufficient_draws", "历史数据少于100期。")
        )

    seen: set[str] = set()
    previous_issue: int | None = None
    previous_date = None
    for index, draw in enumerate(draws):
        try:
            draw.validate()
        except ValueError as exc:
            issues.append(
                StabilityIssue(
                    "critical",
                    "invalid_draw",
                    f"第{index + 1}行数据无效：{exc}",
                )
            )
        if draw.issue in seen:
            issues.append(
                StabilityIssue(
                    "critical",
                    "duplicate_issue",
                    f"期号{draw.issue}重复。",
                )
            )
        seen.add(draw.issue)
        try:
            current_issue = int(draw.issue)
        except ValueError:
            issues.append(
                StabilityIssue(
                    "critical",
                    "non_numeric_issue",
                    f"期号{draw.issue}不是数字。",
                )
            )
            current_issue = None
        if current_issue is not None and previous_issue is not None:
            if current_issue <= previous_issue:
                issues.append(
                    StabilityIssue(
                        "critical",
                        "issue_order",
                        f"期号顺序异常：{previous_issue}之后出现{current_issue}。",
                    )
                )
        if current_issue is not None:
            previous_issue = current_issue
        if draw.draw_date is not None and previous_date is not None:
            if draw.draw_date < previous_date:
                issues.append(
                    StabilityIssue(
                        "warning",
                        "date_order",
                        f"期号{draw.issue}的开奖日期早于上一条记录。",
                    )
                )
        if draw.draw_date is not None:
            previous_date = draw.draw_date

    if not any(item.severity == "critical" for item in issues):
        for zone in ("front", "back"):
            try:
                issue = _prefix_invariance_check(draws, zone)
                if issue is not None:
                    issues.append(issue)
            except Exception as exc:
                issues.append(
                    StabilityIssue(
                        "critical",
                        f"{zone}_audit_failed",
                        f"{zone}区时间泄漏检查失败：{exc}",
                    )
                )

    passed = not any(item.severity == "critical" for item in issues)
    return StabilityAuditResult(
        passed=passed,
        fingerprint=dataset_fingerprint(draws),
        draw_count=len(draws),
        checked_at=datetime.now().isoformat(timespec="seconds"),
        issues=tuple(issues),
    )
