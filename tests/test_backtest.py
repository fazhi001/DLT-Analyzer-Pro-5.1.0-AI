from dlt_analyzer_pro.backtest import rolling_backtest
from dlt_analyzer_pro.importer import load_file
from dlt_analyzer_pro.paths import resource_path


def test_backtest():
    draws, failures = load_file(resource_path("dlt_history.csv"))
    assert not failures
    result = rolling_backtest(draws, periods=3, seed=100)
    assert result.evaluated == 3
    assert len(result.details) == 3
