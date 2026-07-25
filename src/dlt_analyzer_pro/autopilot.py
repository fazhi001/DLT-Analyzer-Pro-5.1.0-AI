from __future__ import annotations

import time
from dataclasses import dataclass, replace

from .ai_engine import AIPredictionSystem
from .ai_settings import dynamic_result_to_dict, save_ai_settings
from .ai_types import AIConfig, AIReport, DynamicWeightResult
from .database import Database
from .dynamic_weight import evaluate_dynamic_weights
from .models import Prediction
from .predictor import next_issue
from .stability import audit_training_pipeline, set_global_seed
from .updater import OfficialDrawUpdater


@dataclass(frozen=True, slots=True)
class AutoPilotResult:
    target_issue: str
    predictions: tuple[Prediction, ...]
    report: AIReport
    dynamic_result: DynamicWeightResult
    update_message: str
    stability_message: str
    backup_path: str
    elapsed_seconds: float


class AIAutoPilot:
    def __init__(self, database: Database, config: AIConfig):
        self.database = database
        self.config = config

    def run(self, target_issue: str | None = None, progress=None) -> AutoPilotResult:
        started = time.perf_counter()
        if self.config.deterministic:
            set_global_seed(self.config.seed)

        backup_path = ""
        if self.config.auto_backup:
            if progress:
                progress("创建并校验数据库备份", 0.01)
            backup = self.database.automatic_backup(
                retention=self.config.backup_retention,
                minimum_interval_hours=24.0,
            )
            if backup is not None:
                backup_path = str(backup)

        update_message = "未执行在线更新"
        if self.config.auto_update:
            if progress:
                progress("检查中国体彩网最新开奖", 0.025)
            try:
                update = OfficialDrawUpdater(self.database).update()
                if update.added or update.updated:
                    update_message = f"官网新增{update.added}期、校正{update.updated}期"
                else:
                    update_message = "官网数据已是最新"
            except Exception as exc:
                update_message = f"在线更新失败，继续使用本地数据：{exc}"

        draws = self.database.all_draws()
        if progress:
            progress("执行数据泄漏与时间完整性检查", 0.05)
        audit = audit_training_pipeline(draws)
        if self.config.leakage_guard and not audit.passed:
            critical = "；".join(
                item.message for item in audit.issues if item.severity == "critical"
            )
            self.database.record_event(
                "stability_block",
                "稳定性检查未通过，已阻止AI训练",
                audit.to_dict(),
            )
            raise RuntimeError(f"稳定性检查未通过：{critical}")
        stability_message = (
            f"稳定性检查通过｜数据指纹{audit.fingerprint[:10]}"
            if audit.passed
            else f"稳定性检查存在{audit.warning_count}项警告"
        )

        if progress:
            progress("滚动回测并评估六个模型", 0.08)
        dynamic = evaluate_dynamic_weights(
            draws,
            periods=self.config.dynamic_periods,
            current_weights=self.config.normalized_weights(),
            learning_rate=self.config.dynamic_learning_rate,
            estimators=self.config.ml_estimators,
            include_ml=True,
            calibrate=self.config.probability_calibration,
            seed=self.config.seed,
            progress=(
                (lambda text, value: progress(text, 0.08 + 0.30 * value))
                if progress else None
            ),
        )
        adaptive_config = replace(
            self.config,
            weights=dict(dynamic.weights),
            front_weights=dict(dynamic.front_weights),
            back_weights=dict(dynamic.back_weights),
            front_model_share=dynamic.front_model_share,
            back_model_share=dynamic.back_model_share,
            baseline_guard=True,
        )
        if progress:
            progress("使用校准概率和模型版本缓存训练", 0.39)
        predictions, report = AIPredictionSystem(adaptive_config).predict(
            draws,
            progress=(
                (lambda text, value: progress(text, 0.39 + 0.60 * value))
                if progress else None
            ),
        )
        report = replace(
            report,
            weight_source="dynamic",
            dynamic_result=dynamic,
            baseline_guard_notes=dynamic.guard_notes,
        )
        issue = (target_issue or "").strip() or next_issue(self.database.latest_issue())
        self.database.save_predictions(issue, predictions)
        ranking_payload = dynamic_result_to_dict(dynamic)["rankings"]
        self.database.save_ai_weight_run(
            issue,
            dynamic.periods,
            dict(dynamic.weights),
            ranking_payload,
            dynamic.confidence,
        )
        self.database.record_event(
            "ai_autopilot",
            f"完成目标期号{issue}的一键自适应预测",
            {
                "fingerprint": audit.fingerprint,
                "seed": adaptive_config.seed,
                "predictions": len(predictions),
                "backup": backup_path,
            },
        )
        save_ai_settings(
            dict(dynamic.weights),
            adaptive_config.simulations,
            adaptive_config.ga_population,
            adaptive_config.ga_generations,
            adaptive_config.ml_estimators,
            dynamic_periods=adaptive_config.dynamic_periods,
            dynamic_learning_rate=adaptive_config.dynamic_learning_rate,
            auto_update=adaptive_config.auto_update,
            last_dynamic_result=dynamic_result_to_dict(dynamic),
            deterministic=adaptive_config.deterministic,
            seed=adaptive_config.seed,
            probability_calibration=adaptive_config.probability_calibration,
            model_cache=adaptive_config.model_cache,
            auto_backup=adaptive_config.auto_backup,
            backup_retention=adaptive_config.backup_retention,
            leakage_guard=adaptive_config.leakage_guard,
            front_weights=dict(dynamic.front_weights),
            back_weights=dict(dynamic.back_weights),
            front_model_share=dynamic.front_model_share,
            back_model_share=dynamic.back_model_share,
            baseline_guard=True,
        )
        if progress:
            progress("可信评估一键预测完成", 1.0)
        return AutoPilotResult(
            target_issue=issue,
            predictions=tuple(predictions),
            report=report,
            dynamic_result=dynamic,
            update_message=update_message,
            stability_message=stability_message,
            backup_path=backup_path,
            elapsed_seconds=time.perf_counter() - started,
        )
