from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.updater import OfficialDrawUpdater


def main() -> int:
    output = PROJECT_ROOT / "resources" / "dlt_history.csv"
    temp_db = PROJECT_ROOT / ".official_history_tmp.db"
    try:
        database = Database(temp_db)
        database.initialize()
        result = OfficialDrawUpdater(database).sync_range("21001", "26081")
        draws = database.all_draws()
        with output.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["issue", "date", "f1", "f2", "f3", "f4", "f5", "b1", "b2"])
            for draw in draws:
                writer.writerow(
                    [
                        draw.issue,
                        draw.draw_date.isoformat() if draw.draw_date else "",
                        *draw.front,
                        *draw.back,
                    ]
                )
        print(
            f"OFFICIAL_HISTORY_OK fetched={result.fetched} "
            f"range={draws[0].issue}-{draws[-1].issue} output={output}"
        )
        return 0
    finally:
        for suffix in ("", "-shm", "-wal"):
            Path(str(temp_db) + suffix).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
