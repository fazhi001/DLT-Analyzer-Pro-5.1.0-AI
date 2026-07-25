from datetime import date

import pytest

from dlt_analyzer_pro.models import Draw


def test_valid_draw():
    Draw("26001", date(2026, 1, 1), (1, 2, 3, 4, 5), (1, 2)).validate()


def test_duplicate_front_rejected():
    with pytest.raises(ValueError):
        Draw("26001", None, (1, 1, 3, 4, 5), (1, 2)).validate()
