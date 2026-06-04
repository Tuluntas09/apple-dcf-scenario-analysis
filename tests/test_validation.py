from datetime import date

import numpy as np
import pandas as pd

from src.validation import concentration_warnings, validate_clean_dataset, validate_requested_inputs


def test_requested_input_rejects_one_asset():
    result = validate_requested_inputs(["SPY"], date(2024, 1, 1), date(2024, 4, 1))

    assert not result.ok
    assert "en az iki varlik" in result.errors[0]


def test_requested_input_warns_on_short_range():
    result = validate_requested_inputs(["SPY", "QQQ"], date(2024, 1, 1), date(2024, 2, 1))

    assert result.ok
    assert result.warnings


def test_clean_dataset_rejects_too_few_returns():
    prices = pd.DataFrame({"SPY": [100, 101], "QQQ": [100, 102]})
    returns = prices.pct_change(fill_method=None).dropna()

    result = validate_clean_dataset(prices, returns)

    assert not result.ok


def test_concentration_warning_detects_large_weight():
    warnings = concentration_warnings(np.array([0.8, 0.2]), ["SPY", "QQQ"])

    assert warnings
