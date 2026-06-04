from __future__ import annotations

import numpy as np
import pandas as pd


def monte_carlo_paths(
    portfolio_returns: pd.Series,
    initial_value: float = 10_000,
    horizon_days: int = 252,
    n_simulations: int = 500,
    seed: int = 11,
) -> pd.DataFrame:
    clean_returns = portfolio_returns.dropna()
    if clean_returns.empty:
        raise ValueError("Portfolio returns are required for simulation.")

    rng = np.random.default_rng(seed)
    mean = clean_returns.mean()
    volatility = clean_returns.std(ddof=1)

    shocks = rng.normal(mean, volatility, size=(horizon_days, n_simulations))
    paths = initial_value * np.cumprod(1 + shocks, axis=0)
    index = pd.RangeIndex(start=1, stop=horizon_days + 1, name="Day")
    columns = [f"Simulation {i + 1}" for i in range(n_simulations)]
    return pd.DataFrame(paths, index=index, columns=columns)


def terminal_value_summary(paths: pd.DataFrame) -> pd.Series:
    terminal_values = paths.iloc[-1]
    return pd.Series(
        {
            "P05": terminal_values.quantile(0.05),
            "P25": terminal_values.quantile(0.25),
            "Median": terminal_values.quantile(0.50),
            "P75": terminal_values.quantile(0.75),
            "P95": terminal_values.quantile(0.95),
        }
    )
