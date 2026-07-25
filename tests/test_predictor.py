from dlt_analyzer_pro.importer import load_file
from dlt_analyzer_pro.paths import resource_path
from dlt_analyzer_pro.predictor import PredictionEngine, STRATEGIES


def test_all_strategies_generate_unique_valid_predictions():
    draws, failures = load_file(resource_path("dlt_history.csv"))
    assert not failures
    for strategy in STRATEGIES:
        predictions = PredictionEngine(seed=42).generate(
            draws,
            count=5,
            strategy=strategy,
            candidate_count=800,
        )
        assert len(predictions) == 5
        assert len({(p.front, p.back) for p in predictions}) == 5
        for prediction in predictions:
            assert len(prediction.front) == 5
            assert len(prediction.back) == 2
