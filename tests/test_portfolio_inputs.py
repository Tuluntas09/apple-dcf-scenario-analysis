import numpy as np
import pandas as pd

from src.portfolio_inputs import lots_to_position_values, lots_to_weights


def test_lots_to_position_values_uses_latest_prices():
    lots = pd.Series({"AAA": 2, "BBB": 3})
    prices = pd.Series({"AAA": 10, "BBB": 20})

    values = lots_to_position_values(lots, prices)

    assert values["AAA"] == 20
    assert values["BBB"] == 60


def test_lots_to_weights_normalizes_position_values():
    lots = pd.Series({"AAA": 2, "BBB": 3})
    prices = pd.Series({"AAA": 10, "BBB": 20})

    weights = lots_to_weights(lots, prices)

    assert np.isclose(weights.sum(), 1)
    assert np.isclose(weights["AAA"], 0.25)
    assert np.isclose(weights["BBB"], 0.75)


def test_lots_to_weights_falls_back_to_equal_weights_when_zero():
    lots = pd.Series({"AAA": 0, "BBB": 0})
    prices = pd.Series({"AAA": 10, "BBB": 20})

    weights = lots_to_weights(lots, prices)

    assert np.isclose(weights["AAA"], 0.5)
    assert np.isclose(weights["BBB"], 0.5)
