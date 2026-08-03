from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .database import Database
from .digit_backtest import rolling_digit_backtest
from .digit_exporter import (
    export_digit_backtest_xlsx,
    export_digit_draws_xlsx,
    export_digit_predictions_xlsx,
)
from .digit_importer import load_digit_file
from .digit_model import (
    GAME_NAMES,
    POSITION_NAMES,
    STRATEGIES,
    DigitModelReport,
    DigitPredictionEngine,
    digit_analysis_rows,
)
from .digit_updater import DigitUpdateResult, OfficialDigitUpdater
from .models import DigitBacktestResult, DigitPrediction
from .predictor import next_issue


class DigitGamesWindow:
    """PL3/PL5 center hosted by the original DLT desktop application."""

    def __init__(self, parent: tk.Tk, database: Database):
        self.parent = parent
        self.database = database
        self.window = tk.Toplevel(parent)
        self.window.title("DLT Analyzer Pro 5.1.1 · 排列三 / 排列五中心")
        self.window.geometry("1180x760")
        self.window.minsize(980, 650)
        self.window.transient(parent)

        self.game_var = tk.StringVar(value="排列三")
        self.status_var = tk.StringVar(value="就绪")
        self.count_var = tk.IntVar(value=10)
        self.strategy_var = tk.StringVar(value="均衡模式")
        self.target_issue_var = tk.StringVar(value="下一期")
        self.periods_var = tk.IntVar(value=50)
        self.current_predictions: list[DigitPrediction] = []
        self.current_backtest: DigitBacktestResult | None = None
        self.busy = False

        self._build()
        self.refresh_all()
        self.window.after(600, self._initial_sync)

    @property
    def game(self) -> str:
        return "pl3" if self.game_var.get() == "排列三" else "pl5"

    def _build(self) -> None:
        root = ttk.Frame(self.window, padding=16)
        root.pack(fill="both", expand=True)

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", pady=(0, 12))
        ttk.Label(toolbar, text="彩票类型：", font=("Microsoft YaHei UI", 10, "bold")).pack(side="left")
        game_box = ttk.Combobox(
            toolbar,
            textvariable=self.game_var,
            values=("排列三", "排列五"),
            state="readonly",
            width=10,
        )
        game_box.pack(side="left", padx=(4, 12))
        game_box.bind("<<ComboboxSelected>>", lambda _event: self._change_game())
        ttk.Button(toolbar, text="同步最新", command=lambda: self.sync_official(False)).pack(side="left")
        ttk.Button(toolbar, text="一键补齐排列数据", command=lambda: self.sync_official(True)).pack(side="left", padx=8)
        ttk.Button(toolbar, text="导入当前彩种", command=self.import_draws).pack(side="left")
        ttk.Button(toolbar, text="导出当前历史", command=self.export_draws).pack(side="left", padx=8)
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="right")

        summary = ttk.Frame(root)
        summary.pack(fill="x", pady=(0, 12))
        self.count_text = tk.StringVar(value="历史数据：0期")
        self.latest_text = tk.StringVar(value="最新期号：-")
        self.model_text = tk.StringVar(value="模型：统计融合")
        for variable in (self.count_text, self.latest_text, self.model_text):
            panel = ttk.LabelFrame(summary, text="", padding=(16, 10))
            panel.pack(side="left", fill="x", expand=True, padx=(0, 8))
            ttk.Label(panel, textvariable=variable, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self._build_history_tab()
        self._build_analysis_tab()
        self._build_prediction_tab()
        self._build_backtest_tab()

        ttk.Label(
            root,
            text="提示：排列三和排列五共用算法框架，但数据、位置模型和验证结果分别保存；历史统计不保证未来中奖。",
            foreground="#6B7280",
        ).pack(fill="x", pady=(10, 0))

    def _build_history_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="历史开奖")
        self.history_tree = ttk.Treeview(tab, show="headings")
        self.history_tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.history_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.history_tree.configure(yscrollcommand=scrollbar.set)

    def _build_analysis_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="位置分析")
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="策略：").pack(side="left")
        strategy_box = ttk.Combobox(
            controls,
            textvariable=self.strategy_var,
            values=tuple(STRATEGIES),
            state="readonly",
            width=12,
        )
        strategy_box.pack(side="left", padx=(4, 8))
        strategy_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_analysis())
        ttk.Button(controls, text="刷新分析", command=self.refresh_analysis).pack(side="left")
        self.analysis_tree = ttk.Treeview(
            tab,
            columns=("position", "digit", "count", "frequency", "omission", "probability", "mode"),
            show="headings",
        )
        columns = [
            ("position", "位置", 90), ("digit", "数字", 70), ("count", "出现次数", 90),
            ("frequency", "历史频率", 100), ("omission", "当前遗漏", 90),
            ("probability", "综合概率", 100), ("mode", "概率来源", 200),
        ]
        for key, title, width in columns:
            self.analysis_tree.heading(key, text=title)
            self.analysis_tree.column(key, width=width, anchor="center")
        self.analysis_tree.pack(fill="both", expand=True)

    def _build_prediction_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="智能选号")
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="目标期号：").pack(side="left")
        ttk.Entry(controls, textvariable=self.target_issue_var, width=10).pack(side="left", padx=(4, 10))
        ttk.Label(controls, text="注数：").pack(side="left")
        ttk.Spinbox(controls, from_=1, to=200, textvariable=self.count_var, width=6).pack(side="left", padx=(4, 10))
        ttk.Label(controls, text="策略：").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.strategy_var,
            values=tuple(STRATEGIES),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(4, 10))
        ttk.Button(controls, text="训练/验证位置模型", command=self.train_models).pack(side="left")
        ttk.Button(controls, text="生成号码", command=self.generate_predictions).pack(side="left", padx=8)
        ttk.Button(controls, text="导出预测", command=self.export_predictions).pack(side="left")

        self.prediction_tree = ttk.Treeview(
            tab,
            columns=("index", "number", "score", "sum", "span", "shape", "mode"),
            show="headings",
        )
        columns = [
            ("index", "序号", 60), ("number", "推荐号码", 130), ("score", "相对评分", 90),
            ("sum", "和值", 70), ("span", "跨度", 70), ("shape", "形态", 100),
            ("mode", "模型模式", 220),
        ]
        for key, title, width in columns:
            self.prediction_tree.heading(key, text=title)
            self.prediction_tree.column(key, width=width, anchor="center")
        self.prediction_tree.pack(fill="both", expand=True)

        self.model_report = tk.Text(tab, height=7, wrap="word")
        self.model_report.pack(fill="x", pady=(8, 0))
        self.model_report.insert("1.0", "点击“训练/验证位置模型”，程序会分别验证每个位置；未优于统计基线的位置不会启用AI。")
        self.model_report.configure(state="disabled")

    def _build_backtest_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="滚动验证")
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="验证期数：").pack(side="left")
        ttk.Spinbox(controls, from_=10, to=200, textvariable=self.periods_var, width=7).pack(side="left", padx=(4, 10))
        ttk.Label(controls, text="滚动验证采用无未来数据的统计融合；AI可信度见训练报告。").pack(side="left")
        ttk.Button(controls, text="开始滚动验证", command=self.run_backtest).pack(side="left", padx=10)
        ttk.Button(controls, text="导出回测", command=self.export_backtest).pack(side="left")
        self.backtest_summary = tk.StringVar(value="尚未执行回测")
        ttk.Label(tab, textvariable=self.backtest_summary, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        self.backtest_tree = ttk.Treeview(
            tab,
            columns=("issue", "model", "random", "exact_model", "exact_random"),
            show="headings",
        )
        for key, title, width in [
            ("issue", "期号", 100), ("model", "模型命中位置", 140),
            ("random", "随机命中位置", 140), ("exact_model", "模型完整命中", 140),
            ("exact_random", "随机完整命中", 140),
        ]:
            self.backtest_tree.heading(key, text=title)
            self.backtest_tree.column(key, width=width, anchor="center")
        self.backtest_tree.pack(fill="both", expand=True)

    def _configure_history_columns(self) -> None:
        game = self.game
        positions = POSITION_NAMES[game]
        columns = ("issue", "date", *[f"p{i}" for i in range(len(positions))], "number", "sum", "span")
        self.history_tree.configure(columns=columns)
        definitions = [("issue", "期号", 90), ("date", "开奖日期", 110)]
        definitions.extend((f"p{i}", name, 65) for i, name in enumerate(positions))
        definitions.extend([("number", "开奖号码", 120), ("sum", "和值", 70), ("span", "跨度", 70)])
        for key, title, width in definitions:
            self.history_tree.heading(key, text=title)
            self.history_tree.column(key, width=width, anchor="center")

    def _change_game(self) -> None:
        self.current_predictions = []
        self.current_backtest = None
        self.refresh_all()

    def refresh_all(self) -> None:
        self._configure_history_columns()
        draws = self.database.all_digit_draws(self.game)
        self.count_text.set(f"历史数据：{len(draws)}期")
        latest = self.database.latest_digit_issue(self.game)
        self.latest_text.set(f"最新期号：{latest or '-'}")
        self.target_issue_var.set(next_issue(latest))
        self._clear_tree(self.history_tree)
        for draw in reversed(draws[-500:]):
            self.history_tree.insert(
                "", "end",
                values=(
                    draw.issue,
                    draw.draw_date.isoformat() if draw.draw_date else "",
                    *draw.digits,
                    "".join(map(str, draw.digits)),
                    sum(draw.digits),
                    max(draw.digits) - min(draw.digits),
                ),
            )
        self.refresh_analysis()

    def refresh_analysis(self) -> None:
        draws = self.database.all_digit_draws(self.game)
        self._clear_tree(self.analysis_tree)
        if not draws:
            return
        try:
            rows = digit_analysis_rows(draws, self.strategy_var.get())
            mode = rows[0]["mode"] if rows else "统计融合"
            self.model_text.set(f"模型：{mode}")
            for row in rows:
                self.analysis_tree.insert(
                    "", "end",
                    values=(
                        row["position"], row["digit"], row["count"],
                        f'{row["frequency"]:.2%}', row["omission"],
                        f'{row["probability"]:.2%}', row["mode"],
                    ),
                )
        except Exception as exc:
            self.status_var.set(f"分析失败：{exc}")

    def _initial_sync(self) -> None:
        if self.database.digit_draw_count("pl5") == 0:
            self.sync_official(full=True, manual=False)
        else:
            self.sync_official(full=False, manual=False)

    def sync_official(self, full: bool, manual: bool = True) -> None:
        if self.busy:
            if manual:
                messagebox.showinfo("任务进行中", "请等待当前任务完成。", parent=self.window)
            return
        self.busy = True
        self.status_var.set("正在连接中国体彩网并同步排列数据…")

        def worker() -> None:
            try:
                updater = OfficialDigitUpdater(self.database)
                result = updater.sync_all() if full else updater.update()
                self.window.after(0, lambda: self._finish_sync(result, manual))
            except Exception as exc:
                logging.getLogger(__name__).exception("Digit official sync failed")
                self.window.after(0, lambda: self._fail_task(f"同步失败：{exc}", manual))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_sync(self, result: DigitUpdateResult, manual: bool) -> None:
        self.busy = False
        self.refresh_all()
        message = (
            f"排列五新增{result.pl5_added}期、校正{result.pl5_updated}期；"
            f"排列三新增{result.pl3_added}期、校正{result.pl3_updated}期；"
            f"官网最新期号{result.latest_remote_issue or '-'}。"
        )
        self.status_var.set(message)
        if manual:
            messagebox.showinfo("同步完成", message, parent=self.window)

    def import_draws(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window,
            title=f"导入{GAME_NAMES[self.game]}数据",
            filetypes=[("支持的文件", "*.csv *.xlsx *.xlsm"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xlsm")],
        )
        if not path:
            return
        try:
            draws, failures = load_digit_file(Path(path), self.game)
            count = self.database.upsert_digit_draws(draws)
            self.refresh_all()
            messagebox.showinfo("导入完成", f"成功导入或更新{count}行，失败{len(failures)}行。", parent=self.window)
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc), parent=self.window)

    def export_draws(self) -> None:
        draws = self.database.all_digit_draws(self.game)
        if not draws:
            messagebox.showwarning("没有数据", "当前彩种没有可导出的历史数据。", parent=self.window)
            return
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="导出历史数据",
            defaultextension=".xlsx",
            initialfile=f"{GAME_NAMES[self.game]}_历史开奖.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if path:
            export_digit_draws_xlsx(Path(path), draws)
            messagebox.showinfo("导出完成", path, parent=self.window)

    def train_models(self) -> None:
        if self.busy:
            return
        draws = self.database.all_digit_draws(self.game)
        if len(draws) < 80:
            messagebox.showwarning("数据不足", "至少需要80期数据训练位置模型。请先一键补齐排列数据。", parent=self.window)
            return
        self.busy = True
        self.status_var.set("正在按时间顺序训练并验证各位置模型…")

        def worker() -> None:
            try:
                report = DigitPredictionEngine(self.game).train_models(draws, force=True)
                self.window.after(0, lambda: self._finish_training(report))
            except Exception as exc:
                logging.getLogger(__name__).exception("Digit model training failed")
                self.window.after(0, lambda: self._fail_task(f"训练失败：{exc}", True))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_training(self, report: DigitModelReport) -> None:
        self.busy = False
        lines = [f"{GAME_NAMES[report.game]}：{report.enabled_count}/{len(report.statuses)}个位置通过样本外验证"]
        for status in report.statuses:
            metric = ""
            if status.validation_logloss is not None and status.baseline_logloss is not None:
                metric = f"；模型LogLoss={status.validation_logloss:.4f}，基线={status.baseline_logloss:.4f}"
            lines.append(f"{status.position_name}：{'启用' if status.enabled else '停用'}（{status.reason}{metric}）")
        self.model_report.configure(state="normal")
        self.model_report.delete("1.0", "end")
        self.model_report.insert("1.0", "\n".join(lines))
        self.model_report.configure(state="disabled")
        self.status_var.set(lines[0])
        self.refresh_analysis()
        messagebox.showinfo("模型训练完成", lines[0], parent=self.window)

    def generate_predictions(self) -> None:
        draws = self.database.all_digit_draws(self.game)
        if len(draws) < 20:
            messagebox.showwarning("数据不足", "至少需要20期历史数据。", parent=self.window)
            return
        try:
            engine = DigitPredictionEngine(self.game)
            self.current_predictions = engine.generate(
                draws,
                count=self.count_var.get(),
                strategy=self.strategy_var.get(),
                use_ml=True,
            )
            target = self.target_issue_var.get().strip() or next_issue(self.database.latest_digit_issue(self.game))
            self.database.save_digit_predictions(self.game, target, self.current_predictions)
            self._clear_tree(self.prediction_tree)
            for index, item in enumerate(self.current_predictions, start=1):
                repeats = len(item.digits) - len(set(item.digits))
                shape = "全异" if repeats == 0 else "一组重复" if repeats == 1 else "多重重复"
                self.prediction_tree.insert(
                    "", "end",
                    values=(
                        index, item.number_text, f"{item.score:.2f}", sum(item.digits),
                        max(item.digits) - min(item.digits), shape, item.model_mode,
                    ),
                )
            mode = self.current_predictions[0].model_mode if self.current_predictions else "-"
            self.model_text.set(f"模型：{mode}")
            self.status_var.set(f"已生成{len(self.current_predictions)}注{GAME_NAMES[self.game]}号码")
        except Exception as exc:
            messagebox.showerror("生成失败", str(exc), parent=self.window)

    def export_predictions(self) -> None:
        if not self.current_predictions:
            messagebox.showwarning("没有预测", "请先生成号码。", parent=self.window)
            return
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="导出预测",
            defaultextension=".xlsx",
            initialfile=f"{GAME_NAMES[self.game]}_{self.target_issue_var.get()}_预测.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if path:
            export_digit_predictions_xlsx(
                Path(path), self.game, self.target_issue_var.get(), self.current_predictions
            )
            messagebox.showinfo("导出完成", path, parent=self.window)

    def run_backtest(self) -> None:
        if self.busy:
            return
        draws = self.database.all_digit_draws(self.game)
        if len(draws) < 60:
            messagebox.showwarning("数据不足", "至少需要60期数据进行滚动验证。", parent=self.window)
            return
        self.busy = True
        self.status_var.set("正在逐期执行滚动验证…")
        periods = self.periods_var.get()
        strategy = self.strategy_var.get()

        def worker() -> None:
            try:
                result = rolling_digit_backtest(draws, periods, strategy, use_ml=False)
                self.window.after(0, lambda: self._finish_backtest(result))
            except Exception as exc:
                logging.getLogger(__name__).exception("Digit backtest failed")
                self.window.after(0, lambda: self._fail_task(f"回测失败：{exc}", True))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_backtest(self, result: DigitBacktestResult) -> None:
        self.busy = False
        self.current_backtest = result
        self.backtest_summary.set(
            f"验证{result.evaluated}期｜模型平均命中{result.model_average_hits:.3f}位｜"
            f"随机{result.random_average_hits:.3f}位｜完整命中：模型{result.model_exact_hits}次 / 随机{result.random_exact_hits}次"
        )
        self._clear_tree(self.backtest_tree)
        for item in reversed(result.details):
            self.backtest_tree.insert(
                "", "end",
                values=(
                    item.issue, item.model_hits, item.random_hits,
                    "是" if item.exact_model else "否", "是" if item.exact_random else "否",
                ),
            )
        self.status_var.set("滚动验证完成")

    def export_backtest(self) -> None:
        if self.current_backtest is None:
            messagebox.showwarning("没有回测", "请先执行滚动验证。", parent=self.window)
            return
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="导出回测",
            defaultextension=".xlsx",
            initialfile=f"{GAME_NAMES[self.game]}_滚动验证.xlsx",
            filetypes=[("Excel", "*.xlsx")],
        )
        if path:
            export_digit_backtest_xlsx(Path(path), self.current_backtest)
            messagebox.showinfo("导出完成", path, parent=self.window)

    def _fail_task(self, message: str, manual: bool) -> None:
        self.busy = False
        self.status_var.set(message + "；本地数据仍可使用")
        if manual:
            messagebox.showerror("操作失败", message, parent=self.window)

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        children = tree.get_children()
        if children:
            tree.delete(*children)
