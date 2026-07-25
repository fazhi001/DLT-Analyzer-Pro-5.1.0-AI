from __future__ import annotations

import tempfile
from pathlib import Path

from .ai_models import component_scores
from .ai_backtest import walk_forward_ai_backtest
from .backtest import rolling_backtest
from .database import Database
from .dynamic_weight import evaluate_dynamic_weights
from .exporter import export_predictions_xlsx
from .importer import load_file
from .paths import resource_path
from .stability import audit_training_pipeline, set_global_seed
from .predictor import PredictionEngine


def run_selftest() -> None:
    """
    Run an isolated self-test without configuring a file logger.

    On Windows, an open logging file handler prevents TemporaryDirectory
    from deleting its directory. The self-test therefore creates the
    temporary database directly instead of bootstrapping normal app logging.
    """
    with tempfile.TemporaryDirectory(prefix="dlt_analyzer_selftest_") as temp:
        data_dir = Path(temp)

        database = Database(data_dir / "dlt_analyzer_v2.db")
        database.initialize()

        draws, failures = load_file(resource_path("dlt_history.csv"))
        if failures:
            raise RuntimeError(f"历史数据存在无效行：{len(failures)}")
        database.upsert_draws(draws)

        count = database.draw_count()
        if count < 800:
            raise RuntimeError(f"历史数据导入异常：{count}")

        stored_draws = database.all_draws()
        set_global_seed(2026)
        audit = audit_training_pipeline(stored_draws)
        if not audit.passed:
            raise RuntimeError("稳定性审计失败")
        predictions = PredictionEngine(seed=2026).generate(
            stored_draws,
            count=5,
            strategy="均衡模式",
            candidate_count=800,
        )
        if len(predictions) != 5:
            raise RuntimeError("预测生成数量异常")
        if len({(p.front, p.back) for p in predictions}) != 5:
            raise RuntimeError("预测结果存在重复")

        output = data_dir / "predictions.xlsx"
        export_predictions_xlsx(output, "SELFTEST", predictions)
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("XLSX导出失败")

        result = rolling_backtest(
            stored_draws,
            periods=3,
            strategy="均衡模式",
            seed=2026,
        )
        if result.evaluated != 3:
            raise RuntimeError("前向验证异常")

        credible = walk_forward_ai_backtest(
            stored_draws, periods=3, include_ml=False,
            bootstrap_samples=500, random_repeats=1000,
        )
        if credible.front_evaluation is None or credible.back_evaluation is None:
            raise RuntimeError("可信评估结果缺失")
        if not (-10.0 < credible.front_evaluation.brier_skill_score < 10.0):
            raise RuntimeError("Brier Skill Score异常")

        front_components, front_metrics = component_scores(
            stored_draws[-260:],
            "front",
            estimators=30,
            include_ml=True,
            fast_ml=True,
        )
        back_components, back_metrics = component_scores(
            stored_draws[-260:],
            "back",
            estimators=30,
            include_ml=True,
            fast_ml=True,
        )
        if len(front_components["xgboost"]) != 35:
            raise RuntimeError("XGBoost前区评分异常")
        if len(back_components["lightgbm"]) != 12:
            raise RuntimeError("LightGBM后区评分异常")
        if len(front_metrics) != 2 or len(back_metrics) != 2:
            raise RuntimeError("机器学习模型评估异常")

        dynamic = evaluate_dynamic_weights(
            stored_draws,
            periods=3,
            current_weights=None,
            learning_rate=0.35,
            estimators=30,
            include_ml=False,
        )
        if len(dynamic.rankings) != 6 or abs(sum(dynamic.weights.values()) - 1.0) > 1e-6:
            raise RuntimeError("动态权重引擎异常")
        database.save_ai_weight_run(
            "SELFTEST", dynamic.periods, dict(dynamic.weights), [], dynamic.confidence
        )
        if database.latest_ai_weight_run() is None:
            raise RuntimeError("动态权重历史保存异常")

        backup = database.verified_backup(data_dir / "backups", retention=3)
        if not backup.exists() or backup.stat().st_size == 0:
            raise RuntimeError("数据库备份失败")
