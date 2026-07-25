from dlt_analyzer_pro.importer import load_file


def test_csv_import(tmp_path):
    path = tmp_path / "draws.csv"
    path.write_text(
        "期号,前区1,前区2,前区3,前区4,前区5,后区1,后区2\n"
        "26001,1,2,3,4,5,1,2\n",
        encoding="utf-8",
    )
    draws, failures = load_file(path)
    assert len(draws) == 1
    assert not failures
    assert draws[0].front == (1, 2, 3, 4, 5)
