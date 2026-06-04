from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, annualized_return, annualized_volatility, max_drawdown, sharpe_ratio


REBALANCE_FREQUENCIES = {
    "Aylik": "ME",
    "Ceyreklik": "QE",
    "Yillik": "YE",
}

STRESS_WINDOWS = {
    "2024 Bahar Geri Cekilme": ("2024-04-02", "2024-04-23"),
    "2024 Yaz Dalgalanmasi": ("2024-08-06", "2024-08-27"),
    "2024 Sonbahar Rotasyonu": ("2024-10-08", "2024-10-29"),
}


def simulate_rebalanced_returns(
    returns: pd.DataFrame,
    target_weights: np.ndarray | list[float],
    frequency: str,
) -> pd.Series:
    target = np.asarray(target_weights, dtype=float)
    target = target / target.sum()
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError("Unsupported rebalance frequency.")

    weights = target.copy()
    values = []
    portfolio_value = 1.0
    rebalance_period = returns.index.to_period(REBALANCE_FREQUENCIES[frequency][0])
    previous_period = rebalance_period[0]

    for timestamp, row in returns.iterrows():
        current_period = timestamp.to_period(REBALANCE_FREQUENCIES[frequency][0])
        if current_period != previous_period:
            weights = target.copy()
            previous_period = current_period

        period_return = float(np.dot(weights, row.values))
        portfolio_value *= 1 + period_return
        values.append(portfolio_value)

        asset_values = weights * (1 + row.values)
        weights = asset_values / asset_values.sum()

    growth = pd.Series(values, index=returns.index, name=frequency)
    return growth.pct_change(fill_method=None).dropna()


def rebalancing_comparison(
    returns: pd.DataFrame,
    target_weights: np.ndarray | list[float],
) -> pd.DataFrame:
    rows = []
    for label in REBALANCE_FREQUENCIES:
        series = simulate_rebalanced_returns(returns, target_weights, label)
        rows.append(
            {
                "Frekans": label,
                "Yillik Getiri": annualized_return(series),
                "Yillik Risk": annualized_volatility(series),
                "Sharpe": sharpe_ratio(series),
                "Max Dusus": max_drawdown(series),
            }
        )
    return pd.DataFrame(rows).set_index("Frekans")


def rolling_metrics(returns: pd.Series, window: int = 63, risk_free_rate: float = 0.0) -> pd.DataFrame:
    rolling_return = (1 + returns).rolling(window).apply(np.prod, raw=True) ** (TRADING_DAYS / window) - 1
    rolling_volatility = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    rolling_sharpe = ((returns - daily_rf).rolling(window).mean() * TRADING_DAYS) / rolling_volatility
    return pd.DataFrame(
        {
            "Rolling Return": rolling_return,
            "Rolling Volatility": rolling_volatility,
            "Rolling Sharpe": rolling_sharpe,
        }
    ).dropna()


def stress_test_summary(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for name, (start, end) in STRESS_WINDOWS.items():
        window_returns = returns.loc[start:end]
        if window_returns.empty:
            rows.append({"Senaryo": name, "Getiri": np.nan, "Max Dusus": np.nan, "Gozlem": 0})
            continue
        rows.append(
            {
                "Senaryo": name,
                "Getiri": float((1 + window_returns).prod() - 1),
                "Max Dusus": max_drawdown(window_returns),
                "Gozlem": len(window_returns),
            }
        )
    return pd.DataFrame(rows).set_index("Senaryo")


def transaction_cost_impact(
    current_weights: pd.Series,
    target_weights: pd.Series,
    cost_bps: float,
) -> pd.Series:
    aligned = pd.concat([current_weights.rename("Mevcut"), target_weights.rename("Hedef")], axis=1).fillna(0)
    turnover = float((aligned["Hedef"] - aligned["Mevcut"]).abs().sum())
    cost = turnover * cost_bps / 10_000
    return pd.Series(
        {
            "Turnover": turnover,
            "Tahmini Maliyet": cost,
        }
    )
