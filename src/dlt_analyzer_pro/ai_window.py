from __future__ import annotations

import threading
from dataclasses import replace
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__, RELEASE_CHANNEL
from .ai_backtest import optimize_ensemble_weights, walk_forward_ai_backtest
from .ai_engine import AIPredictionSystem
from .ai_features import recent_five_years
from .ai_settings import dynamic_result_to_dict, load_ai_settings, save_ai_settings
from .ai_types import AIConfig, COMPONENT_LABELS, DEFAULT_WEIGHTS
from .autopilot import AIAutoPilot
from .database import Database
from .dynamic_weight import evaluate_dynamic_weights
from .exporter import export_ai_report_xlsx, export_backtest_evaluation_xlsx, export_predictions_xlsx
from .model_registry import ModelRegistry
from .models import Prediction
from .paths import stability_report_path
from .predictor import next_issue
from .stability import atomic_write_json, audit_training_pipeline


class AIStudioWindow:
    def __init__(self, parent: tk.Misc, database: Database):
        self.parent = parent
        self.database = database
        self.window = tk.Toplevel(parent)
        self.window.title(f"大乐透 AI 自适应预测系统 {__version__} {RELEASE_CHANNEL}")
        self.window.geometry("1280x840")
        self.window.minsize(1080, 720)

        saved = load_ai_settings()
        self.current_predictions: list[Prediction] = []
        self.current_report = None
        self.current_dynamic_result = None
        self.current_backtest_result = None
        self.running = False

        self.status_var = tk.StringVar(value="就绪")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.dataset_var = tk.StringVar()
        self.autopilot_summary_var = tk.StringVar(value="一键流程尚未运行")
        self.target_issue_var = tk.StringVar(value=next_issue(self.database.latest_issue()))
        self.count_var = tk.IntVar(value=10)
        self.simulations_var = tk.StringVar(value=f'{int(saved["simulations"]):,}')
        self.ga_population_var = tk.IntVar(value=int(saved["ga_population"]))
        self.ga_generations_var = tk.IntVar(value=int(saved["ga_generations"]))
        self.ml_estimators_var = tk.IntVar(value=int(saved["ml_estimators"]))
        self.dynamic_periods_var = tk.IntVar(value=int(saved.get("dynamic_periods", 30)))
        self.dynamic_learning_rate_var = tk.DoubleVar(value=float(saved.get("dynamic_learning_rate", 0.20)))
        self.saved_front_weights = dict(saved.get("front_weights") or {})
        self.saved_back_weights = dict(saved.get("back_weights") or {})
        self.saved_front_model_share = float(saved.get("front_model_share", 1.0))
        self.saved_back_model_share = float(saved.get("back_model_share", 1.0))
        self.baseline_guard_enabled = bool(saved.get("baseline_guard", True))
        self.auto_update_var = tk.BooleanVar(value=bool(saved.get("auto_update", True)))
        self.deterministic_var = tk.BooleanVar(value=bool(saved.get("deterministic", True)))
        self.seed_var = tk.IntVar(value=int(saved.get("seed", 20260721)))
        self.calibration_var = tk.BooleanVar(value=bool(saved.get("probability_calibration", True)))
        self.model_cache_var = tk.BooleanVar(value=bool(saved.get("model_cache", True)))
        self.auto_backup_var = tk.BooleanVar(value=bool(saved.get("auto_backup", True)))
        self.backup_retention_var = tk.IntVar(value=int(saved.get("backup_retention", 10)))
        self.leakage_guard_var = tk.BooleanVar(value=bool(saved.get("leakage_guard", True)))
        self.stability_summary_var = tk.StringVar(value="尚未执行稳定性检查")
        self.guard_summary_var = tk.StringVar(value="基线保护：尚未滚动评估")
        self.backup_summary_var = tk.StringVar(value="数据库备份状态：待检查")
        self.model_registry = ModelRegistry()
        self.weight_vars = {
            name: tk.DoubleVar(value=float(saved["weights"].get(name, value)))
            for name, value in DEFAULT_WEIGHTS.items()
        }

        self._build()
        self._refresh_dataset_status()
        self._load_saved_ranking(saved.get("last_dynamic_result"))

    def _build(self) -> None:
        outer = ttk.Frame(self.window, padding=14)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="大乐透 AI 自适应预测系统", font=("Microsoft YaHei UI", 20, "bold")).pack(side="left")
        ttk.Label(header, text=f"{__version__} {RELEASE_CHANNEL}", foreground="#2563EB", font=("Microsoft YaHei UI", 11, "bold")).pack(side="left", padx=12)
        ttk.Label(header, textvariable=self.dataset_var, foreground="#6B7280").pack(side="right")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        self.predict_tab = ttk.Frame(notebook, padding=12)
        self.model_tab = ttk.Frame(notebook, padding=12)
        self.backtest_tab = ttk.Frame(notebook, padding=12)
        self.optimize_tab = ttk.Frame(notebook, padding=12)
        self.stability_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.predict_tab, text="一键AI预测")
        notebook.add(self.model_tab, text="模型中心")
        notebook.add(self.backtest_tab, text="可信回测")
        notebook.add(self.optimize_tab, text="高级优化")
        notebook.add(self.stability_tab, text="稳定性中心")
        self._build_prediction_tab()
        self._build_model_tab()
        self._build_backtest_tab()
        self._build_optimize_tab()
        self._build_stability_tab()

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Progressbar(footer, variable=self.progress_var, maximum=100.0).pack(side="left", fill="x", expand=True)
        ttk.Label(footer, textvariable=self.status_var, width=50, anchor="e").pack(side="right", padx=(12, 0))

    def _build_prediction_tab(self) -> None:
        flow = ttk.LabelFrame(self.predict_tab, text="一键自适应流程", padding=12)
        flow.pack(fill="x", pady=(0, 10))
        ttk.Label(flow, text="自动备份 → 泄漏检查 → 自动更新 → 滚动回测 → 概率校准 → 版本化训练 → 百万次模拟 → 保存结果", foreground="#374151").pack(side="left")
        ttk.Label(flow, textvariable=self.autopilot_summary_var, foreground="#2563EB").pack(side="right")

        controls = ttk.LabelFrame(self.predict_tab, text="运行参数", padding=12)
        controls.pack(fill="x", pady=(0, 10))
        fields = [
            ("目标期号", ttk.Entry(controls, textvariable=self.target_issue_var, width=11)),
            ("生成注数", ttk.Spinbox(controls, from_=1, to=50, textvariable=self.count_var, width=8)),
            ("蒙特卡洛", ttk.Combobox(controls, textvariable=self.simulations_var, values=("1,000,000", "1,500,000", "2,000,000", "3,000,000"), state="readonly", width=12)),
            ("遗传种群", ttk.Spinbox(controls, from_=100, to=1000, increment=20, textvariable=self.ga_population_var, width=8)),
            ("遗传代数", ttk.Spinbox(controls, from_=20, to=300, increment=10, textvariable=self.ga_generations_var, width=8)),
            ("提升树", ttk.Spinbox(controls, from_=40, to=500, increment=20, textvariable=self.ml_estimators_var, width=8)),
        ]
        for index, (label, widget) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=0, column=index, padx=5, pady=(0, 4))
            widget.grid(row=1, column=index, padx=5, pady=4)
        self.autopilot_button = ttk.Button(controls, text="一键自适应预测", command=self.run_autopilot)
        self.autopilot_button.grid(row=0, column=6, rowspan=2, padx=(18, 5), sticky="ns")
        self.run_button = ttk.Button(controls, text="按当前权重预测", command=self.run_prediction)
        self.run_button.grid(row=0, column=7, rowspan=2, padx=5, sticky="ns")
        ttk.Button(controls, text="导出报告", command=self.export_predictions).grid(row=0, column=8, rowspan=2, padx=5, sticky="ns")

        self.prediction_tree = ttk.Treeview(self.predict_tab, columns=("index", "front", "back", "score", "strategy"), show="headings")
        for column, title, width in [
            ("index", "序号", 65), ("front", "前区", 340), ("back", "后区", 150), ("score", "相对综合分", 120), ("strategy", "模型", 180)
        ]:
            self.prediction_tree.heading(column, text=title)
            self.prediction_tree.column(column, width=width, anchor="center")
        self.prediction_tree.pack(fill="both", expand=True)

    def _build_model_tab(self) -> None:
        settings = ttk.LabelFrame(self.model_tab, text="动态权重设置", padding=12)
        settings.pack(fill="x", pady=(0, 10))
        ttk.Label(settings, text="滚动验证期数").pack(side="left")
        ttk.Spinbox(settings, from_=3, to=100, textvariable=self.dynamic_periods_var, width=7).pack(side="left", padx=(5, 16))
        ttk.Label(settings, text="学习率").pack(side="left")
        ttk.Spinbox(settings, from_=0.05, to=0.85, increment=0.05, textvariable=self.dynamic_learning_rate_var, width=7).pack(side="left", padx=(5, 16))
        ttk.Checkbutton(settings, text="一键预测前自动更新开奖", variable=self.auto_update_var).pack(side="left")
        self.dynamic_button = ttk.Button(settings, text="滚动评估并自动调权", command=self.run_dynamic_weights)
        self.dynamic_button.pack(side="left", padx=18)
        ttk.Button(settings, text="恢复默认权重", command=self.restore_default_weights).pack(side="left")

        weight_panel = ttk.LabelFrame(self.model_tab, text="当前集成权重", padding=10)
        weight_panel.pack(fill="x", pady=(0, 6))
        for index, name in enumerate(DEFAULT_WEIGHTS):
            ttk.Label(weight_panel, text=COMPONENT_LABELS[name]).grid(row=0, column=index, padx=6, pady=(0, 4))
            ttk.Spinbox(weight_panel, from_=0.0, to=1.0, increment=0.01, textvariable=self.weight_vars[name], width=9).grid(row=1, column=index, padx=6, pady=4)

        ttk.Label(
            self.model_tab,
            textvariable=self.guard_summary_var,
            foreground="#2563EB",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(self.model_tab, text="模型排行榜（滚动样本外表现）", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.ranking_tree = ttk.Treeview(self.model_tab, columns=("rank", "model", "weight", "trend", "fh", "bh", "fb", "bb", "score"), show="headings", height=8)
        columns = [
            ("rank", "排名", 55), ("model", "模型", 160), ("weight", "权重", 90), ("trend", "变化", 85),
            ("fh", "前区命中", 95), ("bh", "后区命中", 95), ("fb", "前区Brier", 105), ("bb", "后区Brier", 105), ("score", "质量分", 90)
        ]
        for col, title, width in columns:
            self.ranking_tree.heading(col, text=title)
            self.ranking_tree.column(col, width=width, anchor="center")
        self.ranking_tree.pack(fill="both", expand=True, pady=(4, 8))

        ttk.Label(self.model_tab, text="最终提升树验证指标", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.metric_tree = ttk.Treeview(self.model_tab, columns=("model", "zone", "rows", "brier", "auc", "calibration", "cache"), show="headings", height=5)
        for column, title, width in [("model", "模型", 130), ("zone", "区域", 70), ("rows", "训练样本", 100), ("brier", "校准Brier", 105), ("auc", "验证AUC", 95), ("calibration", "概率校准", 100), ("cache", "模型状态", 110)]:
            self.metric_tree.heading(column, text=title)
            self.metric_tree.column(column, width=width, anchor="center")
        self.metric_tree.pack(fill="x")

    def _build_backtest_tab(self) -> None:
        controls = ttk.LabelFrame(self.backtest_tab, text="滚动样本外可信评估", padding=12)
        controls.pack(fill="x", pady=(0, 8))
        self.backtest_periods_var = tk.IntVar(value=30)
        self.backtest_bootstrap_var = tk.IntVar(value=2000)
        self.backtest_random_repeats_var = tk.IntVar(value=5000)
        self.backtest_include_ml_var = tk.BooleanVar(value=True)
        ttk.Label(controls, text="回测期数").pack(side="left")
        ttk.Spinbox(controls, from_=3, to=120, textvariable=self.backtest_periods_var, width=7).pack(side="left", padx=(5, 12))
        ttk.Label(controls, text="Bootstrap").pack(side="left")
        ttk.Spinbox(controls, from_=500, to=20000, increment=500, textvariable=self.backtest_bootstrap_var, width=8).pack(side="left", padx=(5, 12))
        ttk.Label(controls, text="随机实验").pack(side="left")
        ttk.Spinbox(controls, from_=1000, to=50000, increment=1000, textvariable=self.backtest_random_repeats_var, width=8).pack(side="left", padx=(5, 12))
        ttk.Checkbutton(controls, text="包含XGBoost/LightGBM", variable=self.backtest_include_ml_var).pack(side="left")
        self.backtest_button = ttk.Button(controls, text="运行可信回测", command=self.run_backtest)
        self.backtest_button.pack(side="left", padx=(14, 6))
        self.backtest_export_button = ttk.Button(controls, text="导出评估", command=self.export_backtest_evaluation, state="disabled")
        self.backtest_export_button.pack(side="left", padx=6)

        summary = ttk.LabelFrame(self.backtest_tab, text="Brier Skill Score、Bootstrap 95%区间与随机基线", padding=10)
        summary.pack(fill="x", pady=(0, 8))
        self.backtest_summary_var = tk.StringVar(value="尚未运行可信评估")
        self.backtest_front_var = tk.StringVar(value="前区：-")
        self.backtest_back_var = tk.StringVar(value="后区：-")
        ttk.Label(summary, textvariable=self.backtest_summary_var, foreground="#2563EB", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")
        ttk.Label(summary, textvariable=self.backtest_front_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(summary, textvariable=self.backtest_back_var).pack(anchor="w", pady=(2, 0))

        self.backtest_tree = ttk.Treeview(
            self.backtest_tab,
            columns=("issue", "mf", "mb", "fb", "fr", "bb", "br"),
            show="headings",
        )
        columns = [
            ("issue", "期号", 90), ("mf", "AI前区命中", 100), ("mb", "AI后区命中", 100),
            ("fb", "前区模型Brier", 120), ("fr", "前区基线Brier", 120),
            ("bb", "后区模型Brier", 120), ("br", "后区基线Brier", 120),
        ]
        for column, title, width in columns:
            self.backtest_tree.heading(column, text=title)
            self.backtest_tree.column(column, width=width, anchor="center")
        self.backtest_tree.pack(fill="both", expand=True)
        ttk.Label(
            self.backtest_tab,
            text="BSS>0代表优于均匀概率基线；95%区间跨越0时，不能认定存在稳定优势。随机p值越小，模型平均命中越难由随机票产生。",
            foreground="#6B7280",
        ).pack(anchor="w", pady=(6, 0))

    def _build_optimize_tab(self) -> None:
        controls = ttk.LabelFrame(self.optimize_tab, text="高级随机权重搜索", padding=12)
        controls.pack(fill="x", pady=(0, 10))
        self.optimize_periods_var = tk.IntVar(value=8)
        self.optimize_trials_var = tk.IntVar(value=80)
        self.optimize_include_ml_var = tk.BooleanVar(value=True)
        ttk.Label(controls, text="验证期数").pack(side="left")
        ttk.Spinbox(controls, from_=3, to=20, textvariable=self.optimize_periods_var, width=8).pack(side="left", padx=(5, 18))
        ttk.Label(controls, text="试验次数").pack(side="left")
        ttk.Spinbox(controls, from_=20, to=500, increment=20, textvariable=self.optimize_trials_var, width=8).pack(side="left", padx=(5, 18))
        ttk.Checkbutton(controls, text="包含机器学习", variable=self.optimize_include_ml_var).pack(side="left")
        self.optimize_button = ttk.Button(controls, text="运行高级优化", command=self.run_optimization)
        self.optimize_button.pack(side="left", padx=18)
        self.optimize_result_var = tk.StringVar(value="动态调权是日常默认；高级优化用于实验比较。")
        ttk.Label(self.optimize_tab, textvariable=self.optimize_result_var, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=10)
        self.optimize_tree = ttk.Treeview(self.optimize_tab, columns=("component", "weight"), show="headings", height=10)
        self.optimize_tree.heading("component", text="模型组件")
        self.optimize_tree.heading("weight", text="优化权重")
        self.optimize_tree.column("component", width=260, anchor="center")
        self.optimize_tree.column("weight", width=180, anchor="center")
        self.optimize_tree.pack(fill="x")

    def _build_stability_tab(self) -> None:
        settings = ttk.LabelFrame(self.stability_tab, text="稳定性保护设置", padding=12)
        settings.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(settings, text="固定随机种子", variable=self.deterministic_var).grid(row=0, column=0, padx=8, pady=4, sticky="w")
        ttk.Label(settings, text="种子").grid(row=0, column=1, padx=(12, 4))
        ttk.Spinbox(settings, from_=1, to=2_147_483_647, textvariable=self.seed_var, width=12).grid(row=0, column=2, padx=4)
        ttk.Checkbutton(settings, text="Platt概率校准", variable=self.calibration_var).grid(row=0, column=3, padx=14, pady=4, sticky="w")
        ttk.Checkbutton(settings, text="模型缓存/增量重训练", variable=self.model_cache_var).grid(row=0, column=4, padx=14, pady=4, sticky="w")
        ttk.Checkbutton(settings, text="数据泄漏阻断", variable=self.leakage_guard_var).grid(row=1, column=0, padx=8, pady=4, sticky="w")
        ttk.Checkbutton(settings, text="每日自动备份", variable=self.auto_backup_var).grid(row=1, column=3, padx=14, pady=4, sticky="w")
        ttk.Label(settings, text="保留份数").grid(row=1, column=1, padx=(12, 4))
        ttk.Spinbox(settings, from_=2, to=50, textvariable=self.backup_retention_var, width=8).grid(row=1, column=2, padx=4)

        actions = ttk.LabelFrame(self.stability_tab, text="检查与恢复", padding=12)
        actions.pack(fill="x", pady=(0, 10))
        self.stability_audit_button = ttk.Button(actions, text="运行稳定性检查", command=self.run_stability_audit)
        self.stability_audit_button.pack(side="left", padx=5)
        self.backup_button = ttk.Button(actions, text="立即创建校验备份", command=self.create_verified_backup)
        self.backup_button.pack(side="left", padx=5)
        self.rollback_button = ttk.Button(actions, text="固定到所选模型版本", command=self.rollback_selected_model)
        self.rollback_button.pack(side="left", padx=5)
        self.unpin_button = ttk.Button(actions, text="恢复自动最新版本", command=self.clear_model_pins)
        self.unpin_button.pack(side="left", padx=5)
        ttk.Button(actions, text="刷新版本列表", command=self.refresh_model_versions).pack(side="left", padx=5)
        ttk.Label(actions, textvariable=self.stability_summary_var, foreground="#2563EB").pack(side="right")

        ttk.Label(self.stability_tab, textvariable=self.backup_summary_var).pack(anchor="w", pady=(0, 8))
        self.model_version_tree = ttk.Treeview(
            self.stability_tab,
            columns=("model", "zone", "issue", "created", "brier", "calibrated", "state"),
            show="headings",
        )
        columns = [
            ("model", "模型", 110), ("zone", "区域", 70), ("issue", "数据截止", 90),
            ("created", "创建时间", 155), ("brier", "验证Brier", 105),
            ("calibrated", "校准Brier", 105), ("state", "状态", 115),
        ]
        for column, title, width in columns:
            self.model_version_tree.heading(column, text=title)
            self.model_version_tree.column(column, width=width, anchor="center")
        self.model_version_tree.pack(fill="both", expand=True)
        self.refresh_model_versions()

    def run_stability_audit(self) -> None:
        if self.running:
            return
        self._set_running(True)
        self.status_var.set("执行数据泄漏与数据库完整性检查")
        def worker() -> None:
            try:
                audit = audit_training_pipeline(self.database.all_draws())
                healthy, detail = self.database.integrity_check()
                atomic_write_json(stability_report_path(), {**audit.to_dict(), "database_integrity": detail})
                text = (
                    f"通过｜{audit.draw_count}期｜指纹{audit.fingerprint[:10]}｜数据库{detail}"
                    if audit.passed and healthy
                    else f"未通过｜严重{audit.critical_count}项｜警告{audit.warning_count}项｜数据库{detail}"
                )
                self.window.after(0, lambda: self.stability_summary_var.set(text))
                if not audit.passed or not healthy:
                    details = "\n".join(item.message for item in audit.issues) or detail
                    self.window.after(0, lambda: messagebox.showwarning("稳定性检查", details))
                self.window.after(0, lambda: self.status_var.set("稳定性检查完成"))
            except Exception as exc:
                self.window.after(0, lambda: messagebox.showerror("稳定性检查失败", str(exc)))
            finally:
                self.window.after(0, lambda: self._set_running(False))
        threading.Thread(target=worker, daemon=True).start()

    def create_verified_backup(self) -> None:
        if self.running:
            return
        try:
            retention = int(self.backup_retention_var.get())
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._set_running(True)
        def worker() -> None:
            try:
                path = self.database.verified_backup(retention=retention)
                text = f"数据库备份状态：已校验 {path.name}"
                self.window.after(0, lambda: self.backup_summary_var.set(text))
                self.window.after(0, lambda: messagebox.showinfo("备份完成", str(path)))
            except Exception as exc:
                self.window.after(0, lambda: messagebox.showerror("备份失败", str(exc)))
            finally:
                self.window.after(0, lambda: self._set_running(False))
        threading.Thread(target=worker, daemon=True).start()

    def refresh_model_versions(self) -> None:
        if not hasattr(self, "model_version_tree"):
            return
        self._clear_tree(self.model_version_tree)
        for item in self.model_registry.list_versions():
            state = "已固定" if item.pinned else "当前最新" if item.active else "历史版本"
            self.model_version_tree.insert(
                "", "end", iid=item.version, values=(
                    item.model_name, "前区" if item.zone == "front" else "后区", item.latest_issue,
                    item.created_at, "-" if item.validation_brier is None else f"{item.validation_brier:.5f}",
                    "-" if item.calibrated_brier is None else f"{item.calibrated_brier:.5f}", state,
                )
            )

    def rollback_selected_model(self) -> None:
        selected = self.model_version_tree.selection()
        if not selected:
            messagebox.showwarning("未选择版本", "请先在版本列表中选择一个模型版本。")
            return
        version = selected[0]
        self.model_registry.pin_version(version)
        self.refresh_model_versions()
        self.status_var.set("已固定模型版本；后续预测将使用该版本，直到恢复自动最新。")

    def clear_model_pins(self) -> None:
        self.model_registry.unpin()
        self.refresh_model_versions()
        self.status_var.set("已恢复自动选择最新匹配模型版本。")

    def _refresh_dataset_status(self) -> None:
        draws = recent_five_years(self.database.all_draws())
        self.dataset_var.set(f"最近五年 {len(draws)} 期｜最新 {self.database.latest_issue() or '-'}")

    def _weights(self) -> dict[str, float]:
        return {name: float(variable.get()) for name, variable in self.weight_vars.items()}

    def _config(self) -> AIConfig:
        simulations = int(self.simulations_var.get().replace(",", ""))
        config = AIConfig(
            simulations=max(1_000_000, simulations),
            ga_population=int(self.ga_population_var.get()),
            ga_generations=int(self.ga_generations_var.get()),
            prediction_count=int(self.count_var.get()),
            ml_estimators=int(self.ml_estimators_var.get()),
            weights=self._weights(),
            front_weights=self.saved_front_weights,
            back_weights=self.saved_back_weights,
            front_model_share=self.saved_front_model_share,
            back_model_share=self.saved_back_model_share,
            baseline_guard=self.baseline_guard_enabled,
            dynamic_periods=int(self.dynamic_periods_var.get()),
            dynamic_learning_rate=float(self.dynamic_learning_rate_var.get()),
            auto_update=bool(self.auto_update_var.get()),
            deterministic=bool(self.deterministic_var.get()),
            seed=int(self.seed_var.get()),
            probability_calibration=bool(self.calibration_var.get()),
            model_cache=bool(self.model_cache_var.get()),
            auto_backup=bool(self.auto_backup_var.get()),
            backup_retention=int(self.backup_retention_var.get()),
            leakage_guard=bool(self.leakage_guard_var.get()),
        )
        save_ai_settings(
            self._weights(), config.simulations, config.ga_population, config.ga_generations, config.ml_estimators,
            dynamic_periods=config.dynamic_periods, dynamic_learning_rate=config.dynamic_learning_rate, auto_update=config.auto_update,
            deterministic=config.deterministic, seed=config.seed, probability_calibration=config.probability_calibration,
            model_cache=config.model_cache, auto_backup=config.auto_backup, backup_retention=config.backup_retention,
            leakage_guard=config.leakage_guard,
            front_weights=self.saved_front_weights,
            back_weights=self.saved_back_weights,
            front_model_share=self.saved_front_model_share,
            back_model_share=self.saved_back_model_share,
            baseline_guard=self.baseline_guard_enabled,
        )
        return config

    def _set_running(self, running: bool) -> None:
        self.running = running
        state = "disabled" if running else "normal"
        buttons = [self.autopilot_button, self.run_button, self.dynamic_button, self.backtest_button, self.optimize_button]
        if hasattr(self, "backtest_export_button"):
            export_state = "disabled" if running or self.current_backtest_result is None else "normal"
            self.backtest_export_button.configure(state=export_state)
        for name in ("stability_audit_button", "backup_button", "rollback_button", "unpin_button"):
            if hasattr(self, name):
                buttons.append(getattr(self, name))
        for button in buttons:
            button.configure(state=state)

    def _progress(self, text: str, value: float) -> None:
        self.window.after(0, lambda: (self.status_var.set(text), self.progress_var.set(value * 100.0)))

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        children = tree.get_children()
        if children:
            tree.delete(*children)

    def run_autopilot(self) -> None:
        if self.running:
            return
        try:
            config = self._config()
            target_issue = self.target_issue_var.get().strip()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._set_running(True)
        self.progress_var.set(0.0)
        self.status_var.set("一键流程已启动，正在创建备份")
        self.autopilot_summary_var.set("正在运行，请观察底部进度；首次训练可能需要数分钟")
        def worker() -> None:
            try:
                result = AIAutoPilot(self.database, config).run(target_issue, self._progress)
                self.window.after(0, lambda: self._display_autopilot(result))
            except Exception as exc:
                self.window.after(0, lambda: self.status_var.set("一键流程失败"))
                self.window.after(0, lambda: messagebox.showerror("一键预测失败", str(exc)))
            finally:
                self.window.after(0, lambda: self._set_running(False))
        threading.Thread(target=worker, daemon=True).start()

    def _display_autopilot(self, result) -> None:
        self.target_issue_var.set(result.target_issue)
        self._display_dynamic(result.dynamic_result)
        self._display_prediction(result.predictions, result.report)
        self.autopilot_summary_var.set(f"{result.update_message}｜{result.stability_message}｜动态回测{result.dynamic_result.periods}期｜{result.elapsed_seconds:.1f}秒")
        self.stability_summary_var.set(result.stability_message)
        if result.backup_path:
            self.backup_summary_var.set(f"数据库备份状态：{Path(result.backup_path).name}")
        self.refresh_model_versions()
        self._refresh_dataset_status()

    def run_prediction(self) -> None:
        if self.running:
            return
        try:
            config = self._config()
            target = self.target_issue_var.get().strip() or "下一期"
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._set_running(True)
        self.progress_var.set(0.0)
        self.status_var.set("当前权重预测已启动")
        def worker() -> None:
            try:
                predictions, report = AIPredictionSystem(config).predict(self.database.all_draws(), progress=self._progress)
                self.database.save_predictions(target, predictions)
                self.window.after(0, lambda: self._display_prediction(predictions, report))
            except Exception as exc:
                self.window.after(0, lambda: messagebox.showerror("AI预测失败", str(exc)))
            finally:
                self.window.after(0, lambda: self._set_running(False))
        threading.Thread(target=worker, daemon=True).start()

    def run_dynamic_weights(self) -> None:
        if self.running:
            return
        try:
            config = self._config()
            target_issue = self.target_issue_var.get().strip() or next_issue(self.database.latest_issue())
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._set_running(True)
        self.progress_var.set(0.0)
        def worker() -> None:
            try:
                result = evaluate_dynamic_weights(
                    self.database.all_draws(), periods=config.dynamic_periods, current_weights=config.normalized_weights(),
                    learning_rate=config.dynamic_learning_rate, estimators=config.ml_estimators, include_ml=True,
                    calibrate=config.probability_calibration, seed=config.seed, progress=self._progress,
                )
                save_ai_settings(
                    dict(result.weights), config.simulations, config.ga_population, config.ga_generations, config.ml_estimators,
                    dynamic_periods=config.dynamic_periods, dynamic_learning_rate=config.dynamic_learning_rate,
                    auto_update=config.auto_update, last_dynamic_result=dynamic_result_to_dict(result),
                    front_weights=dict(result.front_weights), back_weights=dict(result.back_weights),
                    front_model_share=result.front_model_share, back_model_share=result.back_model_share,
                    baseline_guard=True,
                )
                self.database.save_ai_weight_run(
                    target_issue, result.periods, dict(result.weights),
                    dynamic_result_to_dict(result)["rankings"], result.confidence,
                )
                self.window.after(0, lambda: self._display_dynamic(result))
            except Exception as exc:
                self.window.after(0, lambda: messagebox.showerror("动态调权失败", str(exc)))
            finally:
                self.window.after(0, lambda: self._set_running(False))
        threading.Thread(target=worker, daemon=True).start()

    def _display_dynamic(self, result) -> None:
        self.current_dynamic_result = result
        self.saved_front_weights = dict(result.front_weights)
        self.saved_back_weights = dict(result.back_weights)
        self.saved_front_model_share = float(result.front_model_share)
        self.saved_back_model_share = float(result.back_model_share)
        self.baseline_guard_enabled = True
        for name, value in result.weights.items():
            if name in self.weight_vars:
                self.weight_vars[name].set(round(float(value), 4))
        self._clear_tree(self.ranking_tree)
        for rank, item in enumerate(result.rankings, start=1):
            trend = "↑" if item.trend > 0.002 else "↓" if item.trend < -0.002 else "→"
            self.ranking_tree.insert("", "end", values=(
                rank, item.label, f"{item.final_weight:.4f}", f"{trend} {item.trend:+.4f}",
                f"{item.front_hits:.3f}", f"{item.back_hits:.3f}", f"{item.front_brier:.5f}",
                f"{item.back_brier:.5f}", f"{item.quality_score:.1f}",
            ))
        self.guard_summary_var.set(
            f"基线保护：前区模型{result.front_model_share:.0%} / 基线{1-result.front_model_share:.0%}，"
            f"BSS {result.front_bss:+.2%}；后区模型{result.back_model_share:.0%} / "
            f"基线{1-result.back_model_share:.0%}，BSS {result.back_bss:+.2%}"
        )
        self.status_var.set(
            f"基线保护完成：回测{result.periods}期｜前区模型{result.front_model_share:.0%}｜"
            f"后区模型{result.back_model_share:.0%}｜置信度{result.confidence:.1%}"
        )
        self.progress_var.set(100.0)

    def _load_saved_ranking(self, payload) -> None:
        if not isinstance(payload, dict): return
        rankings = payload.get("rankings")
        if not isinstance(rankings, list): return
        front_share = float(payload.get("front_model_share", 1.0))
        back_share = float(payload.get("back_model_share", 1.0))
        front_bss = float(payload.get("front_bss", 0.0))
        back_bss = float(payload.get("back_bss", 0.0))
        self.guard_summary_var.set(
            f"基线保护：前区模型{front_share:.0%} / 基线{1-front_share:.0%}，BSS {front_bss:+.2%}；"
            f"后区模型{back_share:.0%} / 基线{1-back_share:.0%}，BSS {back_bss:+.2%}"
        )
        self._clear_tree(self.ranking_tree)
        for rank, item in enumerate(rankings, start=1):
            trend_value = float(item.get("trend", 0.0))
            trend = "↑" if trend_value > 0.002 else "↓" if trend_value < -0.002 else "→"
            self.ranking_tree.insert("", "end", values=(
                rank, item.get("label", item.get("model_name", "-")), f'{float(item.get("final_weight", 0)):.4f}',
                f"{trend} {trend_value:+.4f}", f'{float(item.get("front_hits", 0)):.3f}', f'{float(item.get("back_hits", 0)):.3f}',
                f'{float(item.get("front_brier", 0)):.5f}', f'{float(item.get("back_brier", 0)):.5f}', f'{float(item.get("quality_score", 0)):.1f}',
            ))

    def _display_prediction(self, predictions, report) -> None:
        self.current_predictions = list(predictions)
        self.current_report = report
        self._clear_tree(self.prediction_tree)
        for index, prediction in enumerate(predictions, start=1):
            self.prediction_tree.insert("", "end", values=(
                index, " ".join(f"{n:02d}" for n in prediction.front), " ".join(f"{n:02d}" for n in prediction.back),
                f"{prediction.score:.2f}", prediction.strategy,
            ))
        self._clear_tree(self.metric_tree)
        for metric in report.model_metrics:
            self.metric_tree.insert("", "end", values=(
                metric.model_name, metric.zone, metric.train_rows,
                f"{metric.validation_brier:.5f}" if metric.validation_brier is not None else "-",
                f"{metric.validation_auc:.4f}" if metric.validation_auc is not None else "-",
                metric.calibration_method,
                metric.cache_status,
            ))
        source = "动态权重" if report.weight_source == "dynamic" else "当前权重"
        self.status_var.set(f"完成：{source}｜{report.simulations:,}次模拟｜{report.elapsed_seconds:.1f}秒")
        self.refresh_model_versions()
        self.progress_var.set(100.0)

    def export_predictions(self) -> None:
        if not self.current_predictions:
            messagebox.showwarning("没有结果", "请先运行AI预测。")
            return
        path = filedialog.asksaveasfilename(title="导出AI报告", defaultextension=".xlsx", filetypes=[("Excel文件", "*.xlsx")], initialfile=f"DLT_{self.target_issue_var.get()}_AI报告.xlsx")
        if not path: return
        if self.current_report is not None:
            export_ai_report_xlsx(Path(path), self.target_issue_var.get().strip() or "下一期", self.current_predictions, self.current_report)
        else:
            export_predictions_xlsx(Path(path), self.target_issue_var.get().strip() or "下一期", self.current_predictions)
        messagebox.showinfo("导出完成", path)

    def run_backtest(self) -> None:
        if self.running:
            return
        try:
            periods = int(self.backtest_periods_var.get())
            bootstrap_samples = int(self.backtest_bootstrap_var.get())
            random_repeats = int(self.backtest_random_repeats_var.get())
            include_ml = bool(self.backtest_include_ml_var.get())
            config = self._config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._set_running(True)
        self.progress_var.set(0.0)
        self.backtest_summary_var.set("可信回测正在运行；滚动训练阶段耗时最长")
        self.backtest_front_var.set("前区：计算中")
        self.backtest_back_var.set("后区：计算中")
        self.status_var.set("开始滚动样本外回测")
        def worker() -> None:
            try:
                result = walk_forward_ai_backtest(
                    self.database.all_draws(),
                    periods=periods,
                    config=config,
                    include_ml=include_ml,
                    progress=self._progress,
                    bootstrap_samples=bootstrap_samples,
                    random_repeats=random_repeats,
                    confidence_level=0.95,
                )
                self.window.after(0, lambda: self._display_backtest(result))
            except Exception as exc:
                self.window.after(0, lambda: self.status_var.set("可信回测失败"))
                self.window.after(0, lambda: messagebox.showerror("可信回测失败", str(exc)))
            finally:
                self.window.after(0, lambda: self._set_running(False))
        threading.Thread(target=worker, daemon=True).start()

    def _display_backtest(self, result) -> None:
        self.current_backtest_result = result
        self._clear_tree(self.backtest_tree)
        for row in reversed(result.details):
            self.backtest_tree.insert(
                "", "end",
                values=(
                    row["issue"], row["model_front_hits"], row["model_back_hits"],
                    f'{float(row.get("model_front_brier", 0)):.5f}',
                    f'{float(row.get("reference_front_brier", 0)):.5f}',
                    f'{float(row.get("model_back_brier", 0)):.5f}',
                    f'{float(row.get("reference_back_brier", 0)):.5f}',
                ),
            )
        front = result.front_evaluation
        back = result.back_evaluation
        if front is not None and back is not None:
            self.backtest_summary_var.set(
                f"{result.evaluated}期｜Bootstrap {result.bootstrap_samples:,}次｜随机基线 {result.random_repeats:,}次"
            )
            self.backtest_front_var.set(
                f"前区：BSS {front.brier_skill_score:+.3%}，95%CI [{front.bss_ci_lower:+.3%}, {front.bss_ci_upper:+.3%}]；"
                f"命中 {front.model_hit_average:.3f} vs 随机 {front.random_hit_average:.3f}，提升 {front.hit_uplift:+.3f}，p={front.random_p_value:.4f}；{front.conclusion}"
            )
            self.backtest_back_var.set(
                f"后区：BSS {back.brier_skill_score:+.3%}，95%CI [{back.bss_ci_lower:+.3%}, {back.bss_ci_upper:+.3%}]；"
                f"命中 {back.model_hit_average:.3f} vs 随机 {back.random_hit_average:.3f}，提升 {back.hit_uplift:+.3f}，p={back.random_p_value:.4f}；{back.conclusion}"
            )
        self.backtest_export_button.configure(state="normal")
        self.status_var.set("可信回测完成")
        self.progress_var.set(100.0)

    def export_backtest_evaluation(self) -> None:
        if self.current_backtest_result is None:
            messagebox.showwarning("没有评估结果", "请先运行可信回测。")
            return
        path = filedialog.asksaveasfilename(
            title="导出可信评估",
            defaultextension=".xlsx",
            filetypes=[("Excel文件", "*.xlsx")],
            initialfile=f"DLT_可信评估_{self.current_backtest_result.evaluated}期.xlsx",
        )
        if not path:
            return
        export_backtest_evaluation_xlsx(Path(path), self.current_backtest_result)
        messagebox.showinfo("导出完成", path)

    def run_optimization(self) -> None:
        if self.running:
            return
        try:
            periods = int(self.optimize_periods_var.get())
            trials = int(self.optimize_trials_var.get())
            include_ml = bool(self.optimize_include_ml_var.get())
            config = self._config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self._set_running(True)
        self.progress_var.set(0.0)
        def worker() -> None:
            try:
                result = optimize_ensemble_weights(
                    self.database.all_draws(), periods=periods, trials=trials, config=config,
                    include_ml=include_ml, progress=self._progress,
                )
                self.window.after(0, lambda: self._display_optimization(result))
            except Exception as exc:
                self.window.after(0, lambda: messagebox.showerror("参数优化失败", str(exc)))
            finally:
                self.window.after(0, lambda: self._set_running(False))
        threading.Thread(target=worker, daemon=True).start()

    def _display_optimization(self, result) -> None:
        for name, value in result.best_weights.items():
            if name in self.weight_vars: self.weight_vars[name].set(round(float(value), 4))
        self._clear_tree(self.optimize_tree)
        for name, value in sorted(result.best_weights.items(), key=lambda item: item[1], reverse=True):
            self.optimize_tree.insert("", "end", values=(COMPONENT_LABELS.get(name, name), f"{value:.4f}"))
        self.optimize_result_var.set(f"样本外Brier目标{result.best_objective:.4f}｜命中{result.front_hits:.3f}/{result.back_hits:.3f}｜Brier{result.front_brier:.4f}/{result.back_brier:.4f}")
        self.saved_front_weights = {}
        self.saved_back_weights = {}
        self.saved_front_model_share = 1.0
        self.saved_back_model_share = 1.0
        config = self._config()
        save_ai_settings(
            self._weights(), config.simulations, config.ga_population, config.ga_generations,
            config.ml_estimators, dynamic_periods=config.dynamic_periods,
            dynamic_learning_rate=config.dynamic_learning_rate, auto_update=config.auto_update,
            front_weights={}, back_weights={}, front_model_share=1.0,
            back_model_share=1.0, baseline_guard=True,
        )
        self.status_var.set("高级优化完成，权重已保存"); self.progress_var.set(100.0)

    def restore_default_weights(self) -> None:
        self.saved_front_weights = {}
        self.saved_back_weights = {}
        self.saved_front_model_share = 1.0
        self.saved_back_model_share = 1.0
        self.baseline_guard_enabled = True
        self.guard_summary_var.set("基线保护：已恢复默认权重，下一次滚动评估后生效")
        for name, value in DEFAULT_WEIGHTS.items():
            self.weight_vars[name].set(value)
        config = self._config()
        save_ai_settings(
            dict(DEFAULT_WEIGHTS), config.simulations, config.ga_population,
            config.ga_generations, config.ml_estimators,
            dynamic_periods=config.dynamic_periods,
            dynamic_learning_rate=config.dynamic_learning_rate,
            auto_update=config.auto_update,
            front_weights={}, back_weights={}, front_model_share=1.0,
            back_model_share=1.0, baseline_guard=True,
        )
        self.status_var.set("已恢复默认权重；建议重新运行滚动评估")
