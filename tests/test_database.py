from datetime import date

from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.models import Draw, Prediction


def test_database_roundtrip(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    draw = Draw("26001", date(2026, 1, 1), (1, 2, 3, 4, 5), (1, 2))
    assert database.upsert_draws([draw]) == 1
    assert database.draw_count() == 1
    assert database.latest_issue() == "26001"
    assert database.all_draws()[0] == draw

    prediction = Prediction((1, 2, 3, 4, 5), (1, 2), 2.5, "均衡模式")
    assert database.save_predictions("26002", [prediction]) == 1
    assert len(database.prediction_rows()) == 1
