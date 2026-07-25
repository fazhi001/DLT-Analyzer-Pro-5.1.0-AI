from __future__ import annotations

import logging
import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Type

from . import __version__
from .paths import app_data_dir


def write_crash_report(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
    *,
    context: str = "unhandled",
    data_dir: Path | None = None,
) -> Path:
    root = Path(data_dir or app_data_dir()) / "crash_reports"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"crash_{stamp}.log"
    content = [
        f"DLT Analyzer Pro {__version__} crash report",
        f"time: {datetime.now().isoformat(timespec='seconds')}",
        f"context: {context}",
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        "",
        "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
    ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def install_crash_hooks(data_dir: Path | None = None) -> None:
    previous_sys_hook = sys.excepthook

    def sys_hook(exc_type, exc_value, exc_traceback):
        try:
            path = write_crash_report(
                exc_type,
                exc_value,
                exc_traceback,
                context="main-thread",
                data_dir=data_dir,
            )
            logging.critical("Unhandled exception. Crash report: %s", path, exc_info=(exc_type, exc_value, exc_traceback))
        finally:
            if previous_sys_hook is not None:
                previous_sys_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = sys_hook

    if hasattr(threading, "excepthook"):
        previous_thread_hook = threading.excepthook

        def thread_hook(args):
            try:
                path = write_crash_report(
                    args.exc_type,
                    args.exc_value,
                    args.exc_traceback,
                    context=f"thread:{getattr(args.thread, 'name', 'unknown')}",
                    data_dir=data_dir,
                )
                logging.critical("Unhandled thread exception. Crash report: %s", path)
            finally:
                if previous_thread_hook is not None:
                    previous_thread_hook(args)

        threading.excepthook = thread_hook


def install_tk_exception_hook(root, data_dir: Path | None = None) -> None:
    def report_callback_exception(exc_type, exc_value, exc_traceback):
        path = write_crash_report(
            exc_type,
            exc_value,
            exc_traceback,
            context="tk-callback",
            data_dir=data_dir,
        )
        logging.critical("Tk callback failed. Crash report: %s", path, exc_info=(exc_type, exc_value, exc_traceback))

    root.report_callback_exception = report_callback_exception
