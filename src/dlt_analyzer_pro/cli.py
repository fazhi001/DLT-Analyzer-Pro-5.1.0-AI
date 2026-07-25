from __future__ import annotations

import argparse
import json
import logging
import tkinter as tk
from tkinter import messagebox

from . import __version__
from .bootstrap import initialize_application
from .crash_reporting import install_crash_hooks, install_tk_exception_hook
from .ai_backtest import walk_forward_ai_backtest
from .model_registry import ModelRegistry
from .selftest import run_selftest
from .stability import audit_training_pipeline
from .ui import DLTApplication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="DLT Analyzer Pro")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--gui-smoke", action="store_true")
    parser.add_argument("--stability-check", action="store_true")
    parser.add_argument("--list-model-versions", action="store_true")
    parser.add_argument("--credible-evaluation-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    install_crash_hooks()
    args = build_parser().parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.self_test:
        try:
            run_selftest()
            print("SELFTEST_OK")
            return 0
        except Exception:
            logging.exception("Self-test failed")
            return 1

    try:
        database = initialize_application()
        if args.stability_check:
            result = audit_training_pipeline(database.all_draws())
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            return 0 if result.passed else 2
        if args.credible_evaluation_check:
            result = walk_forward_ai_backtest(
                database.all_draws(), periods=3, include_ml=False,
                bootstrap_samples=500, random_repeats=1000,
            )
            front = result.front_evaluation
            back = result.back_evaluation
            if front is None or back is None:
                return 3
            print(json.dumps({
                "evaluated": result.evaluated,
                "front_bss": front.brier_skill_score,
                "front_bss_ci": [front.bss_ci_lower, front.bss_ci_upper],
                "front_random_p": front.random_p_value,
                "back_bss": back.brier_skill_score,
                "back_bss_ci": [back.bss_ci_lower, back.bss_ci_upper],
                "back_random_p": back.random_p_value,
            }, ensure_ascii=False, indent=2))
            return 0
        if args.list_model_versions:
            print(json.dumps([
                {
                    "version": item.version,
                    "model": item.model_name,
                    "zone": item.zone,
                    "issue": item.latest_issue,
                    "active": item.active,
                    "pinned": item.pinned,
                    "brier": item.validation_brier,
                }
                for item in ModelRegistry().list_versions()
            ], ensure_ascii=False, indent=2))
            return 0

        root = tk.Tk()
        install_tk_exception_hook(root)
        DLTApplication(root, database)
        if args.gui_smoke:
            root.after(1200, root.destroy)
        root.mainloop()
        return 0
    except Exception as exc:
        logging.exception("Application startup failed")
        try:
            messagebox.showerror(f"DLT Analyzer Pro {__version__}", f"程序启动失败：\n{exc}")
        except Exception:
            pass
        return 1
