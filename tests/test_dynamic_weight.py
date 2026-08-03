from dlt_analyzer_pro.ai_types import DEFAULT_WEIGHTS
from dlt_analyzer_pro.database import Database
from dlt_analyzer_pro.dynamic_weight import (
    _ensemble_uncertainty,
    _recency_weights,
    evaluate_dynamic_weights,
)
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


def test_recency_weights_prioritize_newest_fold():
    weights = _recency_weights(5, 0.90)
    assert abs(weights.sum() - 1.0) < 1e-12
    assert weights[-1] > weights[0]
    assert _recency_weights(5, 1.0).tolist() == [0.2] * 5


def test_model_disagreement_is_reported_as_uncertainty():
    fold = {
        "left": np.array([1.0, 0.0]),
        "right": np.array([0.0, 1.0]),
    }
    assert _ensemble_uncertainty([fold], {"left": 0.5, "right": 0.5}) > 0.4


def test_database_saves_weight_history(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    database.save_ai_weight_run("26082", 8, dict(DEFAULT_WEIGHTS), [{"model": "xgboost"}], 0.7)
    row = database.latest_ai_weight_run()
    assert row is not None
    assert row["target_issue"] == "26082"
    assert row["periods"] == 8
    assert row["weights"]["xgboost"] == DEFAULT_WEIGHTS["xgboost"]
import numpy as np
