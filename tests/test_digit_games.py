from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.digit_backtest import rolling_digit_backtest
from dlt_analyzer_pro.digit_model import DigitPredictionEngine, digit_analysis_rows
from dlt_analyzer_pro.digit_updater import OfficialDigitUpdater, derive_pl3, parse_pl5_payload
from dlt_analyzer_pro.models import DigitDraw


def make_draws(game: str, count: int = 140) -> list[DigitDraw]:
    positions = 3 if game == "pl3" else 5
    start = date(2025, 1, 1)
    draws: list[DigitDraw] = []
    for index in range(count):
        digits = tuple((index * (position + 2) + position * 3 + index // 7) % 10 for position in range(positions))
        draws.append(
            DigitDraw(
                game=game,
                issue=str(25001 + index),
                draw_date=start + timedelta(days=index),
                digits=digits,
            )
        )
    return draws


def test_digit_database_isolated(tmp_path: Path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    pl5 = make_draws("pl5", 8)
    pl3 = [derive_pl3(draw) for draw in pl5]
    assert database.upsert_digit_draws(pl5) == 8
    assert database.upsert_digit_draws(pl3) == 8
    assert database.digit_draw_count("pl5") == 8
    assert database.digit_draw_count("pl3") == 8
    assert database.all_digit_draws("pl3")[0].digits == pl5[0].digits[:3]


def test_official_payload_and_sync(tmp_path: Path):
    payload = {
        "success": True,
        "value": {
            "list": [
                {
                    "lotteryDrawNum": "26194",
                    "lotteryDrawResult": "7 0 1 0 6",
                    "lotteryDrawStatus": 20,
                    "lotteryDrawTime": "2026-07-23",
                }
            ]
        },
    }
    assert parse_pl5_payload(payload)[0].digits == (7, 0, 1, 0, 6)
    database = Database(tmp_path / "test.db")
    database.initialize()
    updater = OfficialDigitUpdater(database, fetcher=lambda _url: payload)
    result = updater.update(max_pages=1)
    assert result.pl5_added == 1
    assert result.pl3_added == 1
    assert database.all_digit_draws("pl3")[0].digits == (7, 0, 1)


def test_digit_prediction_and_analysis(tmp_path: Path):
    draws = make_draws("pl5", 140)
    engine = DigitPredictionEngine("pl5", model_base_dir=tmp_path / "models")
    predictions = engine.generate(draws, count=12, use_ml=False)
    assert len(predictions) == 12
    assert all(len(item.digits) == 5 for item in predictions)
    assert len({item.digits for item in predictions}) == 12
    rows = digit_analysis_rows(draws)
    assert len(rows) == 50


def test_digit_model_training_is_position_specific(tmp_path: Path):
    draws = make_draws("pl3", 150)
    engine = DigitPredictionEngine("pl3", model_base_dir=tmp_path / "models")
    report = engine.train_models(draws, force=True)
    assert len(report.statuses) == 3
    assert {status.position for status in report.statuses} == {0, 1, 2}


def test_digit_rolling_backtest():
    draws = make_draws("pl3", 100)
    result = rolling_digit_backtest(draws, periods=20, use_ml=False)
    assert result.evaluated == 20
    assert len(result.position_model_rates) == 3
    assert 0 <= result.model_average_hits <= 3


def test_digit_importer_position_columns_and_leading_zero(tmp_path: Path):
    from dlt_analyzer_pro.digit_importer import load_digit_file

    pl3_path = tmp_path / "pl3.csv"
    pl3_path.write_text("期号,开奖日期,百位,十位,个位\n26195,2026-07-24,0,2,7\n", encoding="utf-8-sig")
    draws, failures = load_digit_file(pl3_path, "pl3")
    assert not failures
    assert draws[0].digits == (0, 2, 7)

    pl5_path = tmp_path / "pl5.csv"
    pl5_path.write_text("期号,开奖日期,开奖号码\n26195,2026-07-24,02719\n", encoding="utf-8-sig")
    draws, failures = load_digit_file(pl5_path, "pl5")
    assert not failures
    assert draws[0].digits == (0, 2, 7, 1, 9)
