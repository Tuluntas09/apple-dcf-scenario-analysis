from __future__ import annotations

import numpy as np
import pandas as pd


def lots_to_position_values(lots: pd.Series, latest_prices: pd.Series) -> pd.Series:
    aligned = pd.concat([lots.rename("Lots"), latest_prices.rename("Price")], axis=1).fillna(0)
    return (aligned["Lots"].clip(lower=0) * aligned["Price"].clip(lower=0)).rename("Position Value")


def lots_to_weights(lots: pd.Series, latest_prices: pd.Series) -> pd.Series:
    values = lots_to_position_values(lots, latest_prices)
    total = values.sum()
    if np.isclose(total, 0):
        return pd.Series(np.repeat(1 / len(values), len(values)), index=values.index, name="Weight")
    return (values / total).rename("Weight")
