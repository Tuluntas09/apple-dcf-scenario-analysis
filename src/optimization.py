from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS


def annual_inputs(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    return returns.mean() * TRADING_DAYS, returns.cov() * TRADING_DAYS


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    portfolio_return = float(np.dot(weights, expected_returns))
    portfolio_volatility = float(np.sqrt(weights.T @ covariance.values @ weights))
    sharpe = (
        (portfolio_return - risk_free_rate) / portfolio_volatility
        if not np.isclose(portfolio_volatility, 0)
        else np.nan
    )
    return portfolio_return, portfolio_volatility, float(sharpe)


def random_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int = 4000,
    risk_free_rate: float = 0.0,
    seed: int = 7,
) -> pd.DataFrame:
    expected_returns, covariance = annual_inputs(returns)
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_portfolios):
        weights = rng.dirichlet(np.ones(len(expected_returns)))
        annual_return, volatility, sharpe = portfolio_performance(
            weights,
            expected_returns,
            covariance,
            risk_free_rate,
        )
        row = {
            "Annual Return": annual_return,
            "Annual Volatility": volatility,
            "Sharpe": sharpe,
        }
        row.update({asset: weight for asset, weight in zip(expected_returns.index, weights)})
        rows.append(row)

    return pd.DataFrame(rows)


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = "max_sharpe",
    risk_free_rate: float = 0.0,
) -> pd.Series:
    expected_returns, covariance = annual_inputs(returns)
    n_assets = len(expected_returns)
    initial_weights = np.repeat(1 / n_assets, n_assets)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda weights: np.sum(weights) - 1},)

    try:
        from scipy.optimize import minimize
    except Exception:
        portfolios = random_portfolios(returns, n_portfolios=8000, risk_free_rate=risk_free_rate)
        target_column = "Sharpe" if objective == "max_sharpe" else "Annual Volatility"
        selected = (
            portfolios.loc[portfolios[target_column].idxmax()]
            if objective == "max_sharpe"
            else portfolios.loc[portfolios[target_column].idxmin()]
        )
        return selected[expected_returns.index].astype(float)

    def negative_sharpe(weights: np.ndarray) -> float:
        _, _, sharpe = portfolio_performance(
            weights,
            expected_returns,
            covariance,
            risk_free_rate,
        )
        return -sharpe

    def volatility(weights: np.ndarray) -> float:
        return portfolio_performance(weights, expected_returns, covariance, risk_free_rate)[1]

    target = negative_sharpe if objective == "max_sharpe" else volatility
    result = minimize(target, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints)

    if not result.success:
        return pd.Series(initial_weights, index=expected_returns.index, name=objective)

    return pd.Series(result.x, index=expected_returns.index, name=objective)
