from __future__ import annotations

from datetime import datetime, timezone

from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.models import Prediction
from dlt_analyzer_pro.time_utils import format_database_timestamp_beijing


def test_utc_database_time_is_displayed_as_beijing_time() -> None:
    assert format_database_timestamp_beijing("2026-07-22 10:18:40") == "2026-07-22 18:18:40"
    assert format_database_timestamp_beijing("2026-07-22T10:18:40+00:00") == "2026-07-22 18:18:40"


def test_prediction_filter_and_delete(tmp_path) -> None:
    database = Database(tmp_path / "history.db")
    database.initialize()
    rows = [
        Prediction((1, 2, 3, 4, 5), (1, 2), 70.0, "AI集成模型"),
        Prediction((2, 3, 4, 5, 6), (3, 4), 80.0, "AI集成模型"),
        Prediction((3, 4, 5, 6, 7), (5, 6), 90.0, "AI集成模型"),
    ]
    database.save_predictions("26082", rows)
    database.save_predictions("26083", [Prediction((4, 5, 6, 7, 8), (7, 8), 95.0, "AI集成模型")])

    assert database.prediction_count() == 4
    assert database.prediction_count(target_issue="26082") == 3
    assert database.prediction_count(min_score=80.0, max_score=90.0) == 2

    filtered = database.prediction_rows(target_issue="26082", min_score=79.0)
    assert [float(row["score"]) for row in filtered] == [90.0, 80.0]

    deleted = database.delete_predictions([int(filtered[0]["id"])])
    assert deleted == 1
    assert database.prediction_count() == 3

    cleared = database.delete_all_predictions()
    assert cleared == 3
    assert database.prediction_count() == 0


def test_realtime_beijing_clock_ticks_and_accepts_network_time() -> None:
    from datetime import timedelta

    from dlt_analyzer_pro.time_utils import NetworkTimeResult, RealtimeBeijingClock

    ticks = iter((100.0, 101.25, 200.0, 201.5))
    clock = RealtimeBeijingClock(
        initial_utc=datetime(2026, 7, 22, 10, 0, 0, tzinfo=timezone.utc),
        monotonic_fn=lambda: next(ticks),
    )
    assert clock.utc_now() == datetime(2026, 7, 22, 10, 0, 1, 250000, tzinfo=timezone.utc)

    clock.apply_network_time(
        NetworkTimeResult(
            utc_time=datetime(2026, 7, 22, 11, 0, 0, tzinfo=timezone.utc),
            source="test-server",
            round_trip_seconds=0.2,
        )
    )
    assert clock.network_synced is True
    assert clock.source_label() == "网络校时"
    assert clock.utc_now() == datetime(2026, 7, 22, 11, 0, 1, 500000, tzinfo=timezone.utc)


def test_network_time_source_fallback() -> None:
    from dlt_analyzer_pro.time_utils import NetworkTimeResult, synchronize_network_time

    calls: list[str] = []

    def fake_fetcher(url: str, *, timeout: float):
        calls.append(url)
        if url == "first":
            raise RuntimeError("offline")
        return NetworkTimeResult(
            utc_time=datetime(2026, 7, 22, 10, 0, 0, tzinfo=timezone.utc),
            source=url,
            round_trip_seconds=0.1,
        )

    result = synchronize_network_time(("first", "second"), fetcher=fake_fetcher)
    assert calls == ["first", "second"]
    assert result.source == "second"
