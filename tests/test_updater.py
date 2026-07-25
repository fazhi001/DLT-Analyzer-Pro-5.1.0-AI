from datetime import date
from urllib.parse import parse_qs, urlparse

from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.models import Draw
from dlt_analyzer_pro.updater import (
    OfficialDrawUpdater,
    build_api_url,
    parse_api_payload,
    parse_api_record,
)


def record(issue, result, draw_time):
    return {
        "lotteryDrawNum": issue,
        "lotteryDrawResult": result,
        "lotteryDrawStatus": 20,
        "lotteryDrawTime": draw_time,
    }


def payload(records):
    return {"success": True, "value": {"list": records}}


def test_build_api_url_points_to_official_history_service():
    url = build_api_url(3, 100)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.netloc == "webapi.sporttery.cn"
    assert query["gameNo"] == ["85"]
    assert query["pageNo"] == ["3"]
    assert query["pageSize"] == ["100"]


def test_parse_api_record():
    draw = parse_api_record(
        record("26081", "08 16 18 24 34 09 12", "2026-07-20")
    )
    assert draw.issue == "26081"
    assert draw.draw_date == date(2026, 7, 20)
    assert draw.front == (8, 16, 18, 24, 34)
    assert draw.back == (9, 12)


def test_parse_api_payload_ignores_invalid_rows():
    draws = parse_api_payload(
        payload(
            [
                record("26081", "08 16 18 24 34 09 12", "2026-07-20"),
                record("26080", "bad data", "2026-07-18"),
            ]
        )
    )
    assert [draw.issue for draw in draws] == ["26081"]


def test_incremental_update(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.upsert_draws(
        [Draw("26080", date(2026, 7, 18), (5, 10, 15, 21, 23), (7, 8))]
    )

    pages = {
        1: payload(
            [
                record("26081", "08 16 18 24 34 09 12", "2026-07-20"),
                record("26080", "05 10 15 21 23 07 08", "2026-07-18"),
            ]
        )
    }

    def fetcher(url):
        page = int(parse_qs(urlparse(url).query)["pageNo"][0])
        return pages.get(page, payload([]))

    result = OfficialDrawUpdater(database, fetcher=fetcher).update()
    assert result.added == 1
    assert database.latest_issue() == "26081"


def test_full_sync_overwrites_and_fills_dates(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.upsert_draws(
        [
            Draw("21001", None, (1, 2, 3, 4, 5), (1, 2)),
            Draw("21002", None, (1, 2, 3, 4, 6), (1, 3)),
        ]
    )

    pages = {
        1: payload(
            [
                record("21002", "02 16 26 31 34 09 11", "2021-01-04"),
                record("21001", "02 06 12 19 33 08 09", "2021-01-02"),
            ]
        )
    }

    def fetcher(url):
        page = int(parse_qs(urlparse(url).query)["pageNo"][0])
        return pages.get(page, payload([]))

    result = OfficialDrawUpdater(database, fetcher=fetcher).sync_range(
        "21001", "21002"
    )
    assert result.fetched == 2
    assert result.updated == 2
    draws = database.all_draws()
    assert draws[0].draw_date == date(2021, 1, 2)
    assert draws[0].front == (2, 6, 12, 19, 33)
