from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

from .models import DigitDraw, DigitPrediction, Draw, Prediction
from .paths import app_data_dir, database_path


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS draws (
    issue TEXT PRIMARY KEY,
    draw_date TEXT,
    f1 INTEGER NOT NULL,
    f2 INTEGER NOT NULL,
    f3 INTEGER NOT NULL,
    f4 INTEGER NOT NULL,
    f5 INTEGER NOT NULL,
    b1 INTEGER NOT NULL,
    b2 INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS digit_draws (
    game TEXT NOT NULL,
    issue TEXT NOT NULL,
    draw_date TEXT,
    d1 INTEGER NOT NULL,
    d2 INTEGER NOT NULL,
    d3 INTEGER NOT NULL,
    d4 INTEGER,
    d5 INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(game, issue)
);

CREATE INDEX IF NOT EXISTS idx_digit_draws_game_issue
ON digit_draws(game, issue DESC);

CREATE TABLE IF NOT EXISTS digit_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game TEXT NOT NULL,
    target_issue TEXT NOT NULL,
    strategy TEXT NOT NULL,
    number_text TEXT NOT NULL,
    score REAL NOT NULL,
    model_mode TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_digit_predictions_game_created
ON digit_predictions(game, created_at DESC);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_issue TEXT NOT NULL,
    strategy TEXT NOT NULL,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at
ON predictions(created_at DESC);

CREATE TABLE IF NOT EXISTS ai_weight_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_issue TEXT NOT NULL,
    periods INTEGER NOT NULL,
    weights_json TEXT NOT NULL,
    ranking_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ai_weight_runs_created_at
ON ai_weight_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_events_created_at
ON system_events(created_at DESC);
"""


class Database:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or database_path())

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=20)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)

    def upsert_draws(self, draws: Iterable[Draw]) -> int:
        rows = []
        for draw in draws:
            draw.validate()
            rows.append(
                (
                    draw.issue,
                    draw.draw_date.isoformat() if draw.draw_date else None,
                    *draw.front,
                    *draw.back,
                )
            )
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO draws(issue, draw_date, f1, f2, f3, f4, f5, b1, b2)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(issue) DO UPDATE SET
                    draw_date=excluded.draw_date,
                    f1=excluded.f1,
                    f2=excluded.f2,
                    f3=excluded.f3,
                    f4=excluded.f4,
                    f5=excluded.f5,
                    b1=excluded.b1,
                    b2=excluded.b2
                """,
                rows,
            )
        return len(rows)

    def draw_count(self) -> int:
        with self.connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM draws").fetchone()[0])

    def latest_issue(self) -> str | None:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT issue
                FROM draws
                ORDER BY CAST(issue AS INTEGER) DESC
                LIMIT 1
                """
            ).fetchone()
        return str(row["issue"]) if row else None

    def all_draws(self) -> list[Draw]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM draws ORDER BY CAST(issue AS INTEGER)"
            ).fetchall()
        return [self._row_to_draw(r) for r in rows]

    def recent_draws(self, limit: int = 200) -> list[Draw]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT * FROM draws
                ORDER BY CAST(issue AS INTEGER) DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._row_to_draw(r) for r in rows]

    def upsert_digit_draws(self, draws: Iterable[DigitDraw]) -> int:
        rows: list[tuple[object, ...]] = []
        for draw in draws:
            draw.validate()
            digits = list(draw.digits) + [None] * (5 - len(draw.digits))
            rows.append((
                draw.game.lower(),
                draw.issue,
                draw.draw_date.isoformat() if draw.draw_date else None,
                *digits,
            ))
        if not rows:
            return 0
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO digit_draws(game, issue, draw_date, d1, d2, d3, d4, d5)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(game, issue) DO UPDATE SET
                    draw_date=excluded.draw_date,
                    d1=excluded.d1, d2=excluded.d2, d3=excluded.d3,
                    d4=excluded.d4, d5=excluded.d5
                """,
                rows,
            )
        return len(rows)

    def digit_draw_count(self, game: str) -> int:
        with self.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM digit_draws WHERE game = ?",
                (game.lower(),),
            ).fetchone()
        return int(row[0]) if row else 0

    def latest_digit_issue(self, game: str) -> str | None:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT issue FROM digit_draws
                WHERE game = ?
                ORDER BY CAST(issue AS INTEGER) DESC LIMIT 1
                """,
                (game.lower(),),
            ).fetchone()
        return str(row["issue"]) if row else None

    def all_digit_draws(self, game: str) -> list[DigitDraw]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT * FROM digit_draws WHERE game = ?
                ORDER BY CAST(issue AS INTEGER)
                """,
                (game.lower(),),
            ).fetchall()
        return [self._row_to_digit_draw(row) for row in rows]

    def recent_digit_draws(self, game: str, limit: int = 300) -> list[DigitDraw]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT * FROM digit_draws WHERE game = ?
                ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?
                """,
                (game.lower(), max(1, int(limit))),
            ).fetchall()
        return [self._row_to_digit_draw(row) for row in rows]

    def save_digit_predictions(
        self,
        game: str,
        target_issue: str,
        predictions: Iterable[DigitPrediction],
    ) -> int:
        rows = [
            (
                game.lower(), target_issue, item.strategy, item.number_text,
                float(item.score), item.model_mode,
            )
            for item in predictions
        ]
        if not rows:
            return 0
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO digit_predictions(
                    game, target_issue, strategy, number_text, score, model_mode
                ) VALUES(?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def digit_prediction_rows(self, game: str, limit: int = 500) -> list[sqlite3.Row]:
        with self.connect() as con:
            return con.execute(
                """
                SELECT id, game, target_issue, strategy, number_text, score,
                       model_mode, created_at
                FROM digit_predictions WHERE game = ?
                ORDER BY id DESC LIMIT ?
                """,
                (game.lower(), max(1, int(limit))),
            ).fetchall()

    def save_predictions(
        self,
        target_issue: str,
        predictions: Iterable[Prediction],
    ) -> int:
        rows = [
            (
                target_issue,
                p.strategy,
                " ".join(f"{n:02d}" for n in p.front),
                " ".join(f"{n:02d}" for n in p.back),
                float(p.score),
            )
            for p in predictions
        ]
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO predictions(target_issue, strategy, front, back, score)
                VALUES(?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def prediction_rows(
        self,
        limit: int = 300,
        target_issue: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        parameters: list[object] = []
        if target_issue:
            clauses.append("target_issue = ?")
            parameters.append(str(target_issue).strip())
        if min_score is not None:
            clauses.append("score >= ?")
            parameters.append(float(min_score))
        if max_score is not None:
            clauses.append("score <= ?")
            parameters.append(float(max_score))

        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(max(1, int(limit)))
        with self.connect() as con:
            return con.execute(
                f"""
                SELECT id, target_issue, strategy, front, back, score, created_at
                FROM predictions
                {where_sql}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(parameters),
            ).fetchall()

    def prediction_count(
        self,
        target_issue: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> int:
        clauses: list[str] = []
        parameters: list[object] = []
        if target_issue:
            clauses.append("target_issue = ?")
            parameters.append(str(target_issue).strip())
        if min_score is not None:
            clauses.append("score >= ?")
            parameters.append(float(min_score))
        if max_score is not None:
            clauses.append("score <= ?")
            parameters.append(float(max_score))
        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.connect() as con:
            row = con.execute(
                f"SELECT COUNT(*) FROM predictions{where_sql}",
                tuple(parameters),
            ).fetchone()
        return int(row[0]) if row else 0

    def delete_predictions(self, prediction_ids: Iterable[int]) -> int:
        ids = sorted({int(item) for item in prediction_ids if int(item) > 0})
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as con:
            cursor = con.execute(
                f"DELETE FROM predictions WHERE id IN ({placeholders})",
                tuple(ids),
            )
            deleted = int(cursor.rowcount if cursor.rowcount is not None else 0)
        self.record_event(
            "prediction_history_delete",
            f"已删除 {deleted} 条预测记录",
            {"prediction_ids": ids},
        )
        return deleted

    def delete_all_predictions(self) -> int:
        with self.connect() as con:
            cursor = con.execute("DELETE FROM predictions")
            deleted = int(cursor.rowcount if cursor.rowcount is not None else 0)
        self.record_event(
            "prediction_history_clear",
            f"已清空 {deleted} 条预测记录",
        )
        return deleted

    def save_ai_weight_run(
        self,
        target_issue: str,
        periods: int,
        weights: dict[str, float],
        rankings: list[dict[str, object]],
        confidence: float,
    ) -> int:
        with self.connect() as con:
            cursor = con.execute(
                """
                INSERT INTO ai_weight_runs(
                    target_issue, periods, weights_json, ranking_json, confidence
                ) VALUES(?,?,?,?,?)
                """,
                (
                    target_issue,
                    int(periods),
                    json.dumps(weights, ensure_ascii=False),
                    json.dumps(rankings, ensure_ascii=False),
                    float(confidence),
                ),
            )
            return int(cursor.lastrowid)

    def latest_ai_weight_run(self) -> dict[str, object] | None:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT target_issue, periods, weights_json, ranking_json,
                       confidence, created_at
                FROM ai_weight_runs
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "target_issue": str(row["target_issue"]),
            "periods": int(row["periods"]),
            "weights": json.loads(row["weights_json"]),
            "rankings": json.loads(row["ranking_json"]),
            "confidence": float(row["confidence"]),
            "created_at": str(row["created_at"]),
        }

    def integrity_check(self) -> tuple[bool, str]:
        with self.connect() as con:
            row = con.execute("PRAGMA quick_check").fetchone()
        detail = str(row[0]) if row else "no result"
        return detail.lower() == "ok", detail

    def record_event(
        self,
        event_type: str,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> int:
        with self.connect() as con:
            cursor = con.execute(
                """
                INSERT INTO system_events(event_type, message, payload_json)
                VALUES(?,?,?)
                """,
                (
                    str(event_type),
                    str(message),
                    json.dumps(payload, ensure_ascii=False) if payload is not None else None,
                ),
            )
            return int(cursor.lastrowid)

    def list_backups(self, target_dir: Path | None = None) -> list[Path]:
        directory = Path(target_dir or (app_data_dir() / "backups"))
        if not directory.exists():
            return []
        return sorted(directory.glob("dlt_analyzer_*.db"), key=lambda item: item.stat().st_mtime, reverse=True)

    def verified_backup(
        self,
        target_dir: Path | None = None,
        retention: int = 10,
    ) -> Path:
        target = self.backup(target_dir)
        con = sqlite3.connect(target)
        try:
            row = con.execute("PRAGMA quick_check").fetchone()
            detail = str(row[0]) if row else "no result"
        finally:
            con.close()
        if detail.lower() != "ok":
            target.unlink(missing_ok=True)
            raise RuntimeError(f"数据库备份校验失败：{detail}")
        backups = self.list_backups(target.parent)
        for obsolete in backups[max(2, int(retention)):]:
            obsolete.unlink(missing_ok=True)
        self.record_event("database_backup", f"数据库备份完成：{target.name}")
        return target

    def automatic_backup(
        self,
        target_dir: Path | None = None,
        retention: int = 10,
        minimum_interval_hours: float = 24.0,
    ) -> Path | None:
        directory = Path(target_dir or (app_data_dir() / "backups"))
        backups = self.list_backups(directory)
        if backups:
            age = datetime.now().timestamp() - backups[0].stat().st_mtime
            if age < max(0.0, float(minimum_interval_hours)) * 3600.0:
                return None
        return self.verified_backup(directory, retention=retention)

    def restore_backup(self, backup_path: Path) -> Path:
        source = Path(backup_path)
        if not source.exists():
            raise FileNotFoundError(source)
        check = sqlite3.connect(source)
        try:
            row = check.execute("PRAGMA quick_check").fetchone()
            detail = str(row[0]) if row else "no result"
        finally:
            check.close()
        if detail.lower() != "ok":
            raise RuntimeError(f"备份文件损坏：{detail}")
        rollback = self.verified_backup(self.path.parent / "backups", retention=10)
        temporary = self.path.with_suffix(".restore.tmp")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, self.path)
        self.record_event("database_restore", f"已恢复备份：{source.name}", {"rollback": str(rollback)})
        return rollback

    def backup(self, target_dir: Path | None = None) -> Path:
        target_dir = Path(target_dir or (app_data_dir() / "backups"))
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = target_dir / f"dlt_analyzer_4_1_{stamp}.db"
        source_con = sqlite3.connect(self.path)
        target_con = sqlite3.connect(target)
        try:
            source_con.backup(target_con)
        finally:
            source_con.close()
            target_con.close()
        return target

    @staticmethod
    def _row_to_digit_draw(row: sqlite3.Row) -> DigitDraw:
        draw_date = None
        if row["draw_date"]:
            draw_date = datetime.strptime(row["draw_date"], "%Y-%m-%d").date()
        game = str(row["game"]).lower()
        count = 3 if game == "pl3" else 5
        digits = tuple(int(row[f"d{index}"]) for index in range(1, count + 1))
        return DigitDraw(
            game=game, issue=str(row["issue"]), draw_date=draw_date, digits=digits
        )

    @staticmethod
    def _row_to_draw(row: sqlite3.Row) -> Draw:
        draw_date = None
        if row["draw_date"]:
            draw_date = datetime.strptime(row["draw_date"], "%Y-%m-%d").date()
        return Draw(
            issue=str(row["issue"]),
            draw_date=draw_date,
            front=(
                int(row["f1"]),
                int(row["f2"]),
                int(row["f3"]),
                int(row["f4"]),
                int(row["f5"]),
            ),
            back=(int(row["b1"]), int(row["b2"])),
        )
