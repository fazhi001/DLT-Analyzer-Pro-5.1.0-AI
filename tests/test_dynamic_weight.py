from dlt_analyzer_pro.ai_types import DEFAULT_WEIGHTS
from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.dynamic_weight import evaluate_dynamic_weights
from dlt_analyzer_pro.importer import load_file
from dlt_analyzer_pro.paths import resource_path


def history():
    draws, failures = load_file(resource_path("dlt_history.csv"))
    assert not failures
    return draws


def test_dynamic_weights_are_normalized_and_ranked():
    result = evaluate_dynamic_weights(
        history(),
        periods=3,
        current_weights=DEFAULT_WEIGHTS,
        learning_rate=0.35,
        estimators=30,
        include_ml=False,
    )
    assert result.periods == 3
    assert len(result.rankings) == 6
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    assert all(0.03 <= value <= 0.43 for value in result.weights.values())


def test_database_saves_weight_history(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.save_ai_weight_run("26082", 8, dict(DEFAULT_WEIGHTS), [{"model": "xgboost"}], 0.7)
    row = database.latest_ai_weight_run()
    assert row is not None
    assert row["target_issue"] == "26082"
    assert row["periods"] == 8
    assert row["weights"]["xgboost"] == DEFAULT_WEIGHTS["xgboost"]
