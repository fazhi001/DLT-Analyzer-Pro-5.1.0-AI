from __future__ import annotations

import logging
from pathlib import Path

from .ai_settings import load_ai_settings
from .database import Database
from .importer import load_file
from .logging_setup import configure_logging
from .paths import ensure_app_dirs, resource_path


def initialize_application(data_dir: Path | None = None) -> Database:
    ensure_app_dirs(data_dir)
    configure_logging(data_dir)
    database = Database(
        (ensure_app_dirs(data_dir) / "dlt_analyzer_v2.db")
        if data_dir is not None
        else None
    )
    database.initialize()

    healthy, detail = database.integrity_check()
    if not healthy:
        raise RuntimeError(f"数据库完整性检查失败：{detail}")

    if database.draw_count() == 0:
        source = resource_path("dlt_history.csv")
        draws, failures = load_file(source)
        database.upsert_draws(draws)
        logging.getLogger(__name__).info(
            "Initial dataset imported: %s rows, %s failures",
            len(draws),
            len(failures),
        )

    settings = load_ai_settings()
    if bool(settings.get("auto_backup", True)) and database.draw_count() > 0:
        try:
            database.automatic_backup(
                retention=int(settings.get("backup_retention", 10)),
                minimum_interval_hours=24.0,
            )
        except Exception:
            logging.getLogger(__name__).exception("Automatic database backup failed")
    return database
