from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__, RELEASE_CHANNEL
from .ai_window import AIStudioWindow
from .analytics import analysis_rows, draw_metrics, summary
from .backtest import rolling_backtest
from .database import Database
from .digit_ui import DigitGamesWindow
from .digit_updater import DigitUpdateResult, OfficialDigitUpdater
from .exporter import export_draws_xlsx, export_predictions_xlsx
from .importer import load_file
from .models import Prediction
from .paths import app_data_dir
from .predictor import PredictionEngine, STRATEGIES, next_issue
from .updater import OfficialDrawUpdater, UpdateError, UpdateResult
from .time_utils import (
    RealtimeBeijingClock,
    format_database_timestamp_beijing,
    synchronize_network_time,
)


BG = "#F3F5F8"
PANEL = "#FFFFFF"
SIDEBAR = "#172033"
TEXT = "#1B2430"
MUTED = "#6B7280"
ACCENT = "#2563EB"
ACCENT_DARK = "#1D4ED8"
BORDER = "#DCE2EA"
RED = "#DC2626"
BLUE = "#2563EB"
AUTO_UPDATE_INTERVAL_MS = 6 * 60 * 60 * 1000
TIME_SYNC_INTERVAL_MS = 30 * 60 * 1000


class DLTApplication:
    def __init__(self, root: tk.Tk, database: Database):
        self.root = root
        self.database = database
        self.current_predictions: list[Prediction] = []
        self.pages: dict[str, ttk.Frame] = {}
        self.update_in_progress = False
        self.auto_update_timer: str | None = None
        self.beijing_clock_timer: str | None = None
        self.beijing_time_sync_timer: str | None = None
        self.beijing_time_sync_in_progress = False
        self.beijing_clock = RealtimeBeijingClock()
        self.digit_games_window: DigitGamesWindow | None = None

        self.root.title(f"DLT Analyzer Pro {__version__} 三彩彩票可信分析系统")
        self.root.geometry("1220x780")
        self.root.minsize(1020, 680)
        self.root.configure(bg=BG)

        self._configure_style()
        self._build_shell()
        self._build_dashboard()
        self._build_history()
        self._build_analysis()
        self._build_prediction()
        self._build_backtest()
        self._build_prediction_history()
        self.show_page("dashboard")
        self.refresh_all()
        self._update_beijing_clock()
        self.root.after(500, self._sync_beijing_time)
        self.root.after(1600, self._initial_official_sync)

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Microsoft YaHei UI", 21, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Microsoft YaHei UI", 10))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT, font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("CardValue.TLabel", background=PANEL, foreground=TEXT, font=("Microsoft YaHei UI", 24, "bold"))
        style.configure("CardLabel.TLabel", background=PANEL, foreground=MUTED, font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Microsoft YaHei UI", 10, "bold"), padding=(16, 9))
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)])
        style.configure("Secondary.TButton", padding=(14, 8))
        style.configure("Treeview", rowheight=30, background=PANEL, fieldbackground=PANEL, bordercolor=BORDER)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"), padding=(6, 8))
        style.map("Treeview", background=[("selected", "#DBEAFE")], foreground=[("selected", TEXT)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 9), font=("Microsoft YaHei UI", 10))
        style.configure("TLabelframe", background=PANEL, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=PANEL, foreground=TEXT, font=("Microsoft YaHei UI", 10, "bold"))

    def _build_shell(self) -> None:
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True)

        sidebar = tk.Frame(shell, bg=SIDEBAR, width=250)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="LOTTERY\nANALYZER PRO",
            bg=SIDEBAR,
            fg="white",
            font=("Microsoft YaHei UI", 18, "bold"),
            justify="left",
        ).pack(anchor="w", padx=24, pady=(26, 28))

        buttons = [
            ("dashboard", "首页概览"),
            ("history", "历史开奖"),
            ("analysis", "号码分析"),
            ("prediction", "智能选号"),
            ("backtest", "前向验证"),
            ("prediction_history", "预测记录"),
        ]
        self.nav_buttons: dict[str, tk.Button] = {}
        for key, label in buttons:
            button = tk.Button(
                sidebar,
                text=label,
                command=lambda k=key: self.show_page(k),
                relief="flat",
                borderwidth=0,
                anchor="w",
                padx=24,
                pady=12,
                bg=SIDEBAR,
                fg="#CBD5E1",
                activebackground="#263248",
                activeforeground="white",
                font=("Microsoft YaHei UI", 11),
                cursor="hand2",
            )
            button.pack(fill="x")
            self.nav_buttons[key] = button

        ai_button = tk.Button(
            sidebar,
            text="AI自适应系统",
            command=self.open_ai_studio,
            relief="flat",
            borderwidth=0,
            anchor="w",
            padx=24,
            pady=12,
            bg="#1D4ED8",
            fg="white",
            activebackground="#2563EB",
            activeforeground="white",
            font=("Microsoft YaHei UI", 11, "bold"),
            cursor="hand2",
        )
        ai_button.pack(fill="x", pady=(8, 0))

        digit_button = tk.Button(
            sidebar,
            text="排列三 / 排列五中心",
            command=self.open_digit_games,
            relief="flat",
            borderwidth=0,
            anchor="w",
            padx=24,
            pady=12,
            bg="#047857",
            fg="white",
            activebackground="#059669",
            activeforeground="white",
            font=("Microsoft YaHei UI", 11, "bold"),
            cursor="hand2",
        )
        digit_button.pack(fill="x", pady=(6, 0))

        tk.Label(
            sidebar,
            text=f"版本 {__version__}\nAI可信评估{RELEASE_CHANNEL}",
            bg=SIDEBAR,
            fg="#7F8DA3",
            font=("Microsoft YaHei UI", 9),
            justify="left",
        ).pack(side="bottom", anchor="w", padx=24, pady=22)

        main = ttk.Frame(shell, style="App.TFrame", padding=(24, 18, 24, 14))
        main.pack(side="left", fill="both", expand=True)

        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill="x", pady=(0, 16))
        title_group = ttk.Frame(header, style="App.TFrame")
        title_group.pack(side="left")
        self.page_title = ttk.Label(title_group, text="", style="Title.TLabel")
        self.page_title.pack(anchor="w")
        self.page_subtitle = ttk.Label(title_group, text="", style="Sub.TLabel")
        self.page_subtitle.pack(anchor="w", pady=(3, 0))

        status_group = ttk.Frame(header, style="App.TFrame")
        status_group.pack(side="right")
        self.status_var = tk.StringVar(value="正在初始化")
        ttk.Label(status_group, textvariable=self.status_var, style="Sub.TLabel").pack(anchor="e")
        self.beijing_time_var = tk.StringVar(value="北京时间：--")
        ttk.Label(status_group, textvariable=self.beijing_time_var, style="Sub.TLabel").pack(anchor="e", pady=(3, 0))

        self.page_container = ttk.Frame(main, style="App.TFrame")
        self.page_container.pack(fill="both", expand=True)

        self.footer_var = tk.StringVar(value="软件仅用于历史数据分析，不保证未来开奖结果。")
        ttk.Label(main, textvariable=self.footer_var, style="Sub.TLabel").pack(fill="x", pady=(10, 0))

    def _new_page(self, key: str) -> ttk.Frame:
        page = ttk.Frame(self.page_container, style="App.TFrame")
        self.pages[key] = page
        return page

    def _panel(self, parent, padding=16):
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=padding)
        return frame

    def _build_dashboard(self) -> None:
        page = self._new_page("dashboard")
        cards = ttk.Frame(page, style="App.TFrame")
        cards.pack(fill="x")

        self.dashboard_count = tk.StringVar(value="0")
        self.dashboard_latest = tk.StringVar(value="-")
        self.dashboard_sum = tk.StringVar(value="-")
        self.dashboard_strategy = tk.StringVar(value="均衡模式")

        card_specs = [
            ("历史数据", self.dashboard_count, "期"),
            ("最新期号", self.dashboard_latest, ""),
            ("近60期平均和值", self.dashboard_sum, ""),
            ("默认策略", self.dashboard_strategy, ""),
        ]
        for index, (label, variable, suffix) in enumerate(card_specs):
            card = self._panel(cards, 18)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 7, 0 if index == 3 else 7))
            ttk.Label(card, text=label, style="CardLabel.TLabel").pack(anchor="w")
            value_row = ttk.Frame(card, style="Panel.TFrame")
            value_row.pack(anchor="w", pady=(8, 0))
            ttk.Label(value_row, textvariable=variable, style="CardValue.TLabel").pack(side="left")
            if suffix:
                ttk.Label(value_row, text=suffix, style="CardLabel.TLabel").pack(side="left", padx=(5, 0), pady=(10, 0))
            cards.columnconfigure(index, weight=1)

        content = ttk.Frame(page, style="App.TFrame")
        content.pack(fill="both", expand=True, pady=(16, 0))

        recent_panel = self._panel(content)
        recent_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(recent_panel, text="最近开奖", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 10))
        self.dashboard_tree = ttk.Treeview(
            recent_panel,
            columns=("issue", "front", "back", "sum"),
            show="headings",
            height=12,
        )
        for column, title, width in [
            ("issue", "期号", 95),
            ("front", "前区", 245),
            ("back", "后区", 110),
            ("sum", "和值", 70),
        ]:
            self.dashboard_tree.heading(column, text=title)
            self.dashboard_tree.column(column, width=width, anchor="center")
        self.dashboard_tree.pack(fill="both", expand=True)

        action_panel = self._panel(content)
        action_panel.pack(side="left", fill="y", padx=(8, 0))
        ttk.Label(action_panel, text="快捷操作", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 12))
        ttk.Button(action_panel, text="进入一键AI系统", style="Accent.TButton", command=self.open_ai_studio).pack(fill="x", pady=5)
        ttk.Button(action_panel, text="生成 10 注统计号码", style="Secondary.TButton", command=self.quick_predict).pack(fill="x", pady=5)
        ttk.Button(action_panel, text="在线更新开奖", style="Secondary.TButton", command=lambda: self.update_online(manual=True)).pack(fill="x", pady=5)
        ttk.Button(action_panel, text="一键同步三种彩票", style="Accent.TButton", command=self.sync_all_games).pack(fill="x", pady=5)
        ttk.Button(action_panel, text="官网全量校验", style="Secondary.TButton", command=lambda: self.update_online(manual=True, full_sync=True)).pack(fill="x", pady=5)
        ttk.Button(action_panel, text="手动导入数据", style="Secondary.TButton", command=self.import_draws).pack(fill="x", pady=5)
        ttk.Button(action_panel, text="备份数据库", style="Secondary.TButton", command=self.backup_database).pack(fill="x", pady=5)
        ttk.Button(action_panel, text="进入前向验证", style="Secondary.TButton", command=lambda: self.show_page("backtest")).pack(fill="x", pady=5)

        notice = tk.Label(
            action_panel,
            text="分析结果仅反映历史数据。\n复杂模型不等于能预测随机开奖。",
            bg=PANEL,
            fg=MUTED,
            justify="left",
            wraplength=230,
            font=("Microsoft YaHei UI", 9),
        )
        notice.pack(anchor="w", pady=(24, 0))

    def _build_history(self) -> None:
        page = self._new_page("history")
        toolbar = ttk.Frame(page, style="App.TFrame")
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Button(toolbar, text="在线更新开奖", style="Accent.TButton", command=lambda: self.update_online(manual=True)).pack(side="left")
        ttk.Button(toolbar, text="官网校验21001—26081", style="Secondary.TButton", command=lambda: self.update_online(manual=True, full_sync=True)).pack(side="left", padx=8)
        ttk.Button(toolbar, text="导入 CSV / XLSX", style="Secondary.TButton", command=self.import_draws).pack(side="left", padx=8)
        ttk.Button(toolbar, text="导出全部开奖", style="Secondary.TButton", command=self.export_draws).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar, text="备份数据库", style="Secondary.TButton", command=self.backup_database).pack(side="left")
        self.history_count_var = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.history_count_var, style="Sub.TLabel").pack(side="right")

        panel = self._panel(page, 12)
        panel.pack(fill="both", expand=True)
        self.history_tree = ttk.Treeview(
            panel,
            columns=("issue", "date", "front", "back", "sum", "odd", "zones"),
            show="headings",
        )
        for column, title, width in [
            ("issue", "期号", 90),
            ("date", "开奖日期", 110),
            ("front", "前区", 260),
            ("back", "后区", 120),
            ("sum", "和值", 70),
            ("odd", "奇偶", 70),
            ("zones", "分区", 70),
        ]:
            self.history_tree.heading(column, text=title)
            self.history_tree.column(column, width=width, anchor="center")
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_analysis(self) -> None:
        page = self._new_page("analysis")
        controls = ttk.Frame(page, style="App.TFrame")
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(controls, text="统计范围：", style="Sub.TLabel").pack(side="left")
        self.analysis_window_var = tk.StringVar(value="近60期")
        window_box = ttk.Combobox(
            controls,
            textvariable=self.analysis_window_var,
            values=("近30期", "近60期", "近100期", "全部"),
            width=10,
            state="readonly",
        )
        window_box.pack(side="left")
        window_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_analysis())
        ttk.Button(controls, text="刷新分析", style="Secondary.TButton", command=self.refresh_analysis).pack(side="left", padx=8)
        self.analysis_summary_var = tk.StringVar()
        ttk.Label(controls, textvariable=self.analysis_summary_var, style="Sub.TLabel").pack(side="right")

        notebook = ttk.Notebook(page)
        notebook.pack(fill="both", expand=True)

        front_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
        back_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=12)
        notebook.add(front_tab, text="前区 01—35")
        notebook.add(back_tab, text="后区 01—12")

        self.front_analysis_tree = ttk.Treeview(
            front_tab,
            columns=("number", "frequency", "omission", "status"),
            show="headings",
        )
        self.back_analysis_tree = ttk.Treeview(
            back_tab,
            columns=("number", "frequency", "omission", "status"),
            show="headings",
        )
        for tree in (self.front_analysis_tree, self.back_analysis_tree):
            for column, title, width in [
                ("number", "号码", 100),
                ("frequency", "出现次数", 130),
                ("omission", "当前遗漏", 130),
                ("status", "状态", 140),
            ]:
                tree.heading(column, text=title)
                tree.column(column, width=width, anchor="center")
            tree.pack(fill="both", expand=True)

    def _build_prediction(self) -> None:
        page = self._new_page("prediction")
        controls = ttk.LabelFrame(page, text="选号参数", padding=14)
        controls.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="策略：").grid(row=0, column=0, padx=(0, 6), pady=4)
        self.strategy_var = tk.StringVar(value="均衡模式")
        ttk.Combobox(
            controls,
            textvariable=self.strategy_var,
            values=tuple(STRATEGIES.keys()),
            state="readonly",
            width=12,
        ).grid(row=0, column=1, padx=(0, 20), pady=4)

        ttk.Label(controls, text="注数：").grid(row=0, column=2, padx=(0, 6), pady=4)
        self.prediction_count_var = tk.IntVar(value=10)
        ttk.Spinbox(
            controls,
            from_=1,
            to=50,
            textvariable=self.prediction_count_var,
            width=7,
        ).grid(row=0, column=3, padx=(0, 20), pady=4)

        ttk.Label(controls, text="目标期号：").grid(row=0, column=4, padx=(0, 6), pady=4)
        self.target_issue_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.target_issue_var, width=14).grid(row=0, column=5, padx=(0, 20), pady=4)

        ttk.Button(
            controls,
            text="开始生成",
            style="Accent.TButton",
            command=self.generate_predictions,
        ).grid(row=0, column=6, padx=4)
        ttk.Button(
            controls,
            text="导出 XLSX",
            style="Secondary.TButton",
            command=self.export_predictions,
        ).grid(row=0, column=7, padx=4)

        panel = self._panel(page, 12)
        panel.pack(fill="both", expand=True)
        self.prediction_tree = ttk.Treeview(
            panel,
            columns=("index", "front", "back", "score", "strategy"),
            show="headings",
        )
        for column, title, width in [
            ("index", "序号", 70),
            ("front", "前区", 330),
            ("back", "后区", 150),
            ("score", "评分", 110),
            ("strategy", "策略", 130),
        ]:
            self.prediction_tree.heading(column, text=title)
            self.prediction_tree.column(column, width=width, anchor="center")
        self.prediction_tree.pack(fill="both", expand=True)

    def _build_backtest(self) -> None:
        page = self._new_page("backtest")
        controls = ttk.LabelFrame(page, text="验证参数", padding=14)
        controls.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="验证期数：").pack(side="left")
        self.backtest_periods_var = tk.IntVar(value=30)
        ttk.Spinbox(
            controls,
            from_=3,
            to=100,
            textvariable=self.backtest_periods_var,
            width=8,
        ).pack(side="left", padx=(4, 18))

        ttk.Label(controls, text="策略：").pack(side="left")
        self.backtest_strategy_var = tk.StringVar(value="均衡模式")
        ttk.Combobox(
            controls,
            textvariable=self.backtest_strategy_var,
            values=tuple(STRATEGIES.keys()),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(4, 18))

        self.backtest_button = ttk.Button(
            controls,
            text="运行滚动前向验证",
            style="Accent.TButton",
            command=self.run_backtest,
        )
        self.backtest_button.pack(side="left")

        self.backtest_summary_var = tk.StringVar(value="尚未运行")
        ttk.Label(controls, textvariable=self.backtest_summary_var, style="Sub.TLabel").pack(side="right")

        panel = self._panel(page, 12)
        panel.pack(fill="both", expand=True)
        self.backtest_tree = ttk.Treeview(
            panel,
            columns=("issue", "mf", "mb", "rf", "rb"),
            show="headings",
        )
        for column, title, width in [
            ("issue", "期号", 130),
            ("mf", "模型前区命中", 150),
            ("mb", "模型后区命中", 150),
            ("rf", "随机前区命中", 150),
            ("rb", "随机后区命中", 150),
        ]:
            self.backtest_tree.heading(column, text=title)
            self.backtest_tree.column(column, width=width, anchor="center")
        self.backtest_tree.pack(fill="both", expand=True)

    def _build_prediction_history(self) -> None:
        page = self._new_page("prediction_history")

        toolbar = self._panel(page, 12)
        toolbar.pack(fill="x", pady=(0, 10))

        self.prediction_history_issue_var = tk.StringVar(value="")
        self.prediction_history_min_score_var = tk.StringVar(value="")
        self.prediction_history_max_score_var = tk.StringVar(value="")
        self.prediction_history_count_var = tk.StringVar(value="共 0 条")

        ttk.Label(toolbar, text="目标期号", style="CardLabel.TLabel").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.prediction_history_issue_var, width=10).pack(side="left", padx=(6, 14))
        ttk.Label(toolbar, text="最低相对综合分", style="CardLabel.TLabel").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.prediction_history_min_score_var, width=8).pack(side="left", padx=(6, 14))
        ttk.Label(toolbar, text="最高相对综合分", style="CardLabel.TLabel").pack(side="left")
        ttk.Entry(toolbar, textvariable=self.prediction_history_max_score_var, width=8).pack(side="left", padx=(6, 14))

        ttk.Button(
            toolbar,
            text="应用筛选",
            style="Secondary.TButton",
            command=self.apply_prediction_history_filters,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="重置",
            style="Secondary.TButton",
            command=self.reset_prediction_history_filters,
        ).pack(side="left", padx=(0, 14))
        ttk.Button(
            toolbar,
            text="删除选中",
            style="Secondary.TButton",
            command=self.delete_selected_prediction_history,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="清空全部",
            style="Secondary.TButton",
            command=self.clear_all_prediction_history,
        ).pack(side="left")
        ttk.Label(toolbar, textvariable=self.prediction_history_count_var, style="CardLabel.TLabel").pack(side="right")

        hint = ttk.Label(
            page,
            text="生成时间统一按北京时间显示；相对综合分仅用于候选组合内部排序，不代表中奖概率。",
            style="Sub.TLabel",
        )
        hint.pack(fill="x", pady=(0, 8))

        panel = self._panel(page, 12)
        panel.pack(fill="both", expand=True)
        self.prediction_history_tree = ttk.Treeview(
            panel,
            columns=("time", "issue", "front", "back", "score", "strategy"),
            show="headings",
            selectmode="extended",
        )
        for column, title, width in [
            ("time", "生成时间（北京时间）", 180),
            ("issue", "目标期号", 100),
            ("front", "前区", 270),
            ("back", "后区", 120),
            ("score", "相对综合分", 110),
            ("strategy", "策略", 120),
        ]:
            self.prediction_history_tree.heading(column, text=title)
            self.prediction_history_tree.column(column, width=width, anchor="center")
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=self.prediction_history_tree.yview)
        self.prediction_history_tree.configure(yscrollcommand=scrollbar.set)
        self.prediction_history_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def open_ai_studio(self) -> None:
        AIStudioWindow(self.root, self.database)

    def open_digit_games(self) -> None:
        if (
            self.digit_games_window is not None
            and self.digit_games_window.window.winfo_exists()
        ):
            self.digit_games_window.window.deiconify()
            self.digit_games_window.window.lift()
            self.digit_games_window.window.focus_force()
            return
        self.digit_games_window = DigitGamesWindow(self.root, self.database)

    def show_page(self, key: str) -> None:
        titles = {
            "dashboard": ("首页概览", "查看数据状态和最近开奖"),
            "history": ("历史开奖", "导入、浏览、导出和备份开奖数据"),
            "analysis": ("号码分析", "查看频率、冷热状态和当前遗漏"),
            "prediction": ("智能选号", "基于历史统计生成多样化组合"),
            "backtest": ("前向验证", "使用过去数据模拟真实的逐期预测"),
            "prediction_history": ("预测记录", "按北京时间查看、筛选并安全清理历史预测"),
        }
        for page in self.pages.values():
            page.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        self.page_title.configure(text=titles[key][0])
        self.page_subtitle.configure(text=titles[key][1])
        for button_key, button in self.nav_buttons.items():
            button.configure(
                bg="#263248" if button_key == key else SIDEBAR,
                fg="white" if button_key == key else "#CBD5E1",
            )
        if key == "analysis":
            self.refresh_analysis()
        elif key == "prediction_history":
            self.refresh_prediction_history()

    def refresh_all(self) -> None:
        count = self.database.draw_count()
        latest = self.database.latest_issue()
        draws = self.database.all_draws()
        recent = draws[-60:]
        info = summary(recent)

        self.status_var.set(f"数据库 {count} 期｜最新期号 {latest or '-'}")
        self.dashboard_count.set(str(count))
        self.dashboard_latest.set(latest or "-")
        self.dashboard_sum.set(f'{info["average_sum"]:.1f}' if recent else "-")
        self.target_issue_var.set(next_issue(latest))
        self.history_count_var.set(f"共 {count} 期")

        self._fill_dashboard(draws[-10:])
        self._fill_history(self.database.recent_draws(500))
        self.refresh_analysis()
        self.refresh_prediction_history()

    def _fill_dashboard(self, draws) -> None:
        self._clear_tree(self.dashboard_tree)
        for draw in reversed(draws):
            metrics = draw_metrics(draw)
            self.dashboard_tree.insert(
                "",
                "end",
                values=(
                    draw.issue,
                    " ".join(f"{n:02d}" for n in draw.front),
                    " ".join(f"{n:02d}" for n in draw.back),
                    metrics["sum"],
                ),
            )

    def _fill_history(self, draws) -> None:
        self._clear_tree(self.history_tree)
        for draw in draws:
            metrics = draw_metrics(draw)
            self.history_tree.insert(
                "",
                "end",
                values=(
                    draw.issue,
                    draw.draw_date.isoformat() if draw.draw_date else "",
                    " ".join(f"{n:02d}" for n in draw.front),
                    " ".join(f"{n:02d}" for n in draw.back),
                    metrics["sum"],
                    f'{metrics["odd"]}:{metrics["even"]}',
                    ":".join(str(n) for n in metrics["zones"]),
                ),
            )

    def refresh_analysis(self) -> None:
        all_draws = self.database.all_draws()
        selection = self.analysis_window_var.get()
        mapping = {"近30期": 30, "近60期": 60, "近100期": 100}
        draws = all_draws[-mapping[selection]:] if selection in mapping else all_draws
        if not draws:
            return
        front_rows, back_rows = analysis_rows(draws)
        front_freq_values = sorted(row["frequency"] for row in front_rows)
        back_freq_values = sorted(row["frequency"] for row in back_rows)
        front_hot = front_freq_values[-8]
        front_cold = front_freq_values[7]
        back_hot = back_freq_values[-3]
        back_cold = back_freq_values[2]

        self._clear_tree(self.front_analysis_tree)
        for row in front_rows:
            status = "热" if row["frequency"] >= front_hot else "冷" if row["frequency"] <= front_cold else "中性"
            self.front_analysis_tree.insert(
                "",
                "end",
                values=(
                    f'{row["number"]:02d}',
                    row["frequency"],
                    row["omission"],
                    status,
                ),
            )

        self._clear_tree(self.back_analysis_tree)
        for row in back_rows:
            status = "热" if row["frequency"] >= back_hot else "冷" if row["frequency"] <= back_cold else "中性"
            self.back_analysis_tree.insert(
                "",
                "end",
                values=(
                    f'{row["number"]:02d}',
                    row["frequency"],
                    row["omission"],
                    status,
                ),
            )

        info = summary(draws)
        self.analysis_summary_var.set(
            f'{len(draws)}期｜平均和值 {info["average_sum"]:.1f}｜'
            f'平均奇数 {info["average_odd"]:.2f}｜'
            f'含连号比例 {info["consecutive_rate"]:.1%}'
        )

    def quick_predict(self) -> None:
        self.strategy_var.set("均衡模式")
        self.prediction_count_var.set(10)
        self.show_page("prediction")
        self.generate_predictions()

    def generate_predictions(self) -> None:
        try:
            self.footer_var.set("正在生成预测组合…")
            self.root.update_idletasks()
            count = int(self.prediction_count_var.get())
            strategy = self.strategy_var.get()
            engine = PredictionEngine()
            predictions = engine.generate(
                self.database.all_draws(),
                count=count,
                strategy=strategy,
            )
            self.current_predictions = predictions
            target = self.target_issue_var.get().strip() or "下一期"
            self.database.save_predictions(target, predictions)
            self._clear_tree(self.prediction_tree)
            for index, prediction in enumerate(predictions, start=1):
                self.prediction_tree.insert(
                    "",
                    "end",
                    values=(
                        index,
                        " ".join(f"{n:02d}" for n in prediction.front),
                        " ".join(f"{n:02d}" for n in prediction.back),
                        f"{prediction.score:.4f}",
                        prediction.strategy,
                    ),
                )
            self.refresh_prediction_history()
            self.footer_var.set(f"已生成并保存 {len(predictions)} 注组合。")
        except Exception as exc:
            messagebox.showerror("生成失败", str(exc))
            self.footer_var.set("生成失败，请检查数据。")

    def run_backtest(self) -> None:
        self.backtest_button.configure(state="disabled")
        self.backtest_summary_var.set("正在计算，请稍候…")
        self._clear_tree(self.backtest_tree)

        def worker() -> None:
            try:
                result = rolling_backtest(
                    self.database.all_draws(),
                    periods=int(self.backtest_periods_var.get()),
                    strategy=self.backtest_strategy_var.get(),
                )
                self.root.after(0, lambda: self._display_backtest(result))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("验证失败", str(exc)))
            finally:
                self.root.after(0, lambda: self.backtest_button.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _display_backtest(self, result) -> None:
        self.backtest_summary_var.set(
            f"模型 前区 {result.model_front_average:.3f} / 后区 {result.model_back_average:.3f}｜"
            f"随机 前区 {result.random_front_average:.3f} / 后区 {result.random_back_average:.3f}"
        )
        for detail in reversed(result.details):
            self.backtest_tree.insert(
                "",
                "end",
                values=(
                    detail.issue,
                    detail.model_front_hits,
                    detail.model_back_hits,
                    detail.random_front_hits,
                    detail.random_back_hits,
                ),
            )

    @staticmethod
    def _optional_score(text: str, label: str) -> float | None:
        value = str(text).strip()
        if not value:
            return None
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"{label}必须是数字。") from exc
        if number < 0:
            raise ValueError(f"{label}不能小于 0。")
        return number

    def _prediction_history_filters(self) -> tuple[str | None, float | None, float | None]:
        target_issue = self.prediction_history_issue_var.get().strip() or None
        minimum = self._optional_score(
            self.prediction_history_min_score_var.get(),
            "最低相对综合分",
        )
        maximum = self._optional_score(
            self.prediction_history_max_score_var.get(),
            "最高相对综合分",
        )
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("最低相对综合分不能高于最高相对综合分。")
        return target_issue, minimum, maximum

    def apply_prediction_history_filters(self) -> None:
        try:
            self.refresh_prediction_history(strict=True)
        except ValueError as exc:
            messagebox.showwarning("筛选条件无效", str(exc))

    def reset_prediction_history_filters(self) -> None:
        self.prediction_history_issue_var.set("")
        self.prediction_history_min_score_var.set("")
        self.prediction_history_max_score_var.set("")
        self.refresh_prediction_history()

    def refresh_prediction_history(self, strict: bool = False) -> None:
        if not hasattr(self, "prediction_history_tree"):
            return
        try:
            target_issue, minimum, maximum = self._prediction_history_filters()
        except ValueError:
            if strict:
                raise
            self.prediction_history_count_var.set("筛选条件无效，请重置或重新输入")
            return
        rows = self.database.prediction_rows(
            2000,
            target_issue=target_issue,
            min_score=minimum,
            max_score=maximum,
        )
        total = self.database.prediction_count(
            target_issue=target_issue,
            min_score=minimum,
            max_score=maximum,
        )
        self._clear_tree(self.prediction_history_tree)
        for row in rows:
            self.prediction_history_tree.insert(
                "",
                "end",
                iid=str(row["id"]),
                values=(
                    format_database_timestamp_beijing(row["created_at"]),
                    row["target_issue"],
                    row["front"],
                    row["back"],
                    f'{row["score"]:.4f}',
                    row["strategy"],
                ),
            )
        suffix = "（最多显示 2000 条）" if total > 2000 else ""
        self.prediction_history_count_var.set(f"筛选结果 {total} 条{suffix}")

    def delete_selected_prediction_history(self) -> None:
        selected = self.prediction_history_tree.selection()
        if not selected:
            messagebox.showinfo("未选择记录", "请先在表格中选择要删除的预测记录。支持按住 Ctrl 或 Shift 多选。")
            return
        ids = [int(item) for item in selected]
        if not messagebox.askyesno(
            "确认删除",
            f"确定删除选中的 {len(ids)} 条预测记录吗？\n\n删除前会自动创建数据库校验备份，开奖历史和模型不会被删除。",
        ):
            return
        try:
            backup = self.database.verified_backup(retention=10)
            deleted = self.database.delete_predictions(ids)
            self.refresh_prediction_history()
            self.footer_var.set(f"已删除 {deleted} 条预测记录；备份：{backup.name}")
        except Exception as exc:
            messagebox.showerror("删除失败", str(exc))

    def clear_all_prediction_history(self) -> None:
        count = self.database.prediction_count()
        if count <= 0:
            messagebox.showinfo("无需清理", "当前没有预测历史记录。")
            return
        if not messagebox.askyesno(
            "清空全部预测历史",
            f"当前共有 {count} 条预测记录。确定全部清空吗？\n\n此操作只删除预测记录，不删除开奖历史、模型、设置和数据库备份。删除前会自动备份。",
        ):
            return
        if not messagebox.askyesno(
            "二次确认",
            "请再次确认：清空后的预测记录只能通过刚创建的数据库备份恢复。是否继续？",
        ):
            return
        try:
            backup = self.database.verified_backup(retention=10)
            deleted = self.database.delete_all_predictions()
            self.refresh_prediction_history()
            self.footer_var.set(f"已清空 {deleted} 条预测记录；备份：{backup.name}")
        except Exception as exc:
            messagebox.showerror("清空失败", str(exc))

    def _update_beijing_clock(self) -> None:
        if hasattr(self, "beijing_time_var"):
            label = self.beijing_clock.source_label()
            self.beijing_time_var.set(
                f"北京时间（{label}）：{self.beijing_clock.format_now()}"
            )
        # 对齐到下一秒边界，减少长时间运行后的显示漂移。
        microseconds = self.beijing_clock.beijing_now().microsecond
        delay_ms = max(100, 1000 - microseconds // 1000)
        self.beijing_clock_timer = self.root.after(delay_ms, self._update_beijing_clock)

    def _schedule_next_time_sync(self) -> None:
        if self.beijing_time_sync_timer is not None:
            try:
                self.root.after_cancel(self.beijing_time_sync_timer)
            except tk.TclError:
                pass
        self.beijing_time_sync_timer = self.root.after(
            TIME_SYNC_INTERVAL_MS, self._sync_beijing_time
        )

    def _sync_beijing_time(self) -> None:
        if self.beijing_time_sync_in_progress:
            return
        self.beijing_time_sync_in_progress = True

        def worker() -> None:
            try:
                result = synchronize_network_time(timeout=8.0)
                self.root.after(0, lambda value=result: self._finish_time_sync(value))
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Beijing time synchronization failed: %s", exc
                )
                self.root.after(0, self._fail_time_sync)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_time_sync(self, result) -> None:
        self.beijing_time_sync_in_progress = False
        self.beijing_clock.apply_network_time(result)
        self._update_beijing_clock()
        self._schedule_next_time_sync()

    def _fail_time_sync(self) -> None:
        self.beijing_time_sync_in_progress = False
        # 网络不可用时继续按本机时钟实时走秒，不阻塞任何业务。
        self._schedule_next_time_sync()

    def _schedule_next_auto_update(self) -> None:
        if self.auto_update_timer is not None:
            try:
                self.root.after_cancel(self.auto_update_timer)
            except tk.TclError:
                pass
        self.auto_update_timer = self.root.after(
            AUTO_UPDATE_INTERVAL_MS,
            lambda: self.update_online(manual=False),
        )

    def _initial_official_sync(self) -> None:
        draws = self.database.all_draws()
        needs_full_sync = any(draw.draw_date is None for draw in draws)
        self.update_online(manual=False, full_sync=needs_full_sync)

    def update_online(self, manual: bool = True, full_sync: bool = False) -> None:
        if self.update_in_progress:
            if manual:
                messagebox.showinfo("正在更新", "开奖数据更新正在进行，请稍候。")
            return

        self.update_in_progress = True
        action_text = "正在从中国体彩网校验21001—26081…" if full_sync else "正在连接中国体彩网检查新期开奖…"
        self.footer_var.set(action_text)
        self.status_var.set(action_text)

        def worker() -> None:
            try:
                updater = OfficialDrawUpdater(self.database)
                result = updater.sync_range("21001", "26081") if full_sync else updater.update()
                self.root.after(
                    0,
                    lambda value=result: self._finish_online_update(value, manual),
                )
            except Exception as exc:
                logging.getLogger(__name__).exception("Online draw update failed")
                message = str(exc)
                self.root.after(
                    0,
                    lambda value=message: self._fail_online_update(value, manual),
                )

        threading.Thread(target=worker, daemon=True).start()

    def sync_all_games(self) -> None:
        if self.update_in_progress:
            messagebox.showinfo("正在更新", "开奖数据更新正在进行，请稍候。")
            return
        self.update_in_progress = True
        action_text = "正在一键同步大乐透、排列三和排列五…"
        self.footer_var.set(action_text)
        self.status_var.set(action_text)

        def worker() -> None:
            try:
                dlt_updater = OfficialDrawUpdater(self.database)
                local_draws = self.database.all_draws()
                full_result = None
                if any(draw.draw_date is None for draw in local_draws):
                    full_result = dlt_updater.sync_range("21001", "26081")
                incremental_result = dlt_updater.update()

                digit_updater = OfficialDigitUpdater(self.database)
                if self.database.digit_draw_count("pl5") == 0:
                    digit_result = digit_updater.sync_all()
                else:
                    digit_result = digit_updater.update()
                self.root.after(
                    0,
                    lambda: self._finish_all_games_sync(
                        full_result, incremental_result, digit_result
                    ),
                )
            except Exception as exc:
                logging.getLogger(__name__).exception("All games synchronization failed")
                message = str(exc)
                self.root.after(
                    0,
                    lambda: self._fail_online_update(message, True),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _finish_all_games_sync(
        self,
        full_result: UpdateResult | None,
        dlt_result: UpdateResult,
        digit_result: DigitUpdateResult,
    ) -> None:
        self.update_in_progress = False
        self._schedule_next_auto_update()
        self.refresh_all()
        dlt_added = dlt_result.added + (full_result.added if full_result else 0)
        dlt_updated = dlt_result.updated + (full_result.updated if full_result else 0)
        message = (
            f"同步完成：大乐透新增{dlt_added}期、校正{dlt_updated}期；"
            f"排列五新增{digit_result.pl5_added}期、校正{digit_result.pl5_updated}期；"
            f"排列三新增{digit_result.pl3_added}期、校正{digit_result.pl3_updated}期。"
        )
        self.footer_var.set(message)
        messagebox.showinfo("三种彩票同步完成", message)
        if (
            self.digit_games_window is not None
            and self.digit_games_window.window.winfo_exists()
        ):
            self.digit_games_window.refresh_all()

    def _finish_online_update(self, result: UpdateResult, manual: bool) -> None:
        self.update_in_progress = False
        self._schedule_next_auto_update()
        self.refresh_all()
        if result.added:
            message = (
                f"已从{result.source_name}新增 {result.added} 期、校正 {result.updated} 期，"
                f"最新期号 {result.latest_remote_issue}。"
            )
            self.footer_var.set(message)
            if manual:
                messagebox.showinfo("更新完成", message)
        else:
            if result.full_sync:
                message = f"官网全量校验完成：共核对 {result.fetched} 期（21001—26081）。"
            else:
                message = f"已检查中国体彩网，当前已是最新期号 {result.latest_remote_issue or '-'}。"
            self.footer_var.set(message)
            if manual:
                messagebox.showinfo("无需更新", message)

    def _fail_online_update(self, message: str, manual: bool) -> None:
        self.update_in_progress = False
        self._schedule_next_auto_update()
        self.refresh_all()
        display = f"在线更新失败：{message}"
        self.footer_var.set(display + "；本地数据仍可正常使用。")
        if manual:
            messagebox.showerror("在线更新失败", display)

    def import_draws(self) -> None:
        path = filedialog.askopenfilename(
            title="选择开奖数据文件",
            filetypes=[
                ("支持的文件", "*.csv *.xlsx"),
                ("CSV 文件", "*.csv"),
                ("Excel 文件", "*.xlsx"),
            ],
        )
        if not path:
            return
        try:
            draws, failures = load_file(Path(path))
            count = self.database.upsert_draws(draws)
            self.refresh_all()
            messagebox.showinfo(
                "导入完成",
                f"成功导入或更新：{count} 行\n失败：{len(failures)} 行",
            )
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def export_draws(self) -> None:
        path = filedialog.asksaveasfilename(
            title="导出历史开奖",
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx")],
            initialfile="DLT_历史开奖.xlsx",
        )
        if not path:
            return
        try:
            export_draws_xlsx(Path(path), self.database.all_draws())
            messagebox.showinfo("导出完成", path)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def export_predictions(self) -> None:
        if not self.current_predictions:
            messagebox.showwarning("没有预测结果", "请先生成预测。")
            return
        path = filedialog.asksaveasfilename(
            title="导出预测结果",
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx")],
            initialfile=f"DLT_{self.target_issue_var.get()}_预测.xlsx",
        )
        if not path:
            return
        try:
            export_predictions_xlsx(
                Path(path),
                self.target_issue_var.get().strip() or "下一期",
                self.current_predictions,
            )
            messagebox.showinfo("导出完成", path)
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def backup_database(self) -> None:
        try:
            target = self.database.backup(app_data_dir() / "backups")
            messagebox.showinfo("备份完成", str(target))
        except Exception as exc:
            messagebox.showerror("备份失败", str(exc))

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        children = tree.get_children()
        if children:
            tree.delete(*children)
