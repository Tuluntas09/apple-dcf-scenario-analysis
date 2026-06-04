import numpy as np
import pandas as pd

from src.data_loader import load_sample_prices
from src.metrics import calculate_returns, portfolio_metric_summary, portfolio_returns
from src.optimization import optimize_portfolio, portfolio_performance


def test_app_core_calculation_smoke_path():
    prices = load_sample_prices(["SPY", "QQQ", "TLT", "GLD"])
    returns = calculate_returns(prices)
    weights = np.repeat(0.25, 4)
    portfolio = portfolio_returns(returns, weights)
    summary = portfolio_metric_summary(portfolio, risk_free_rate=0.04)
    optimized = optimize_portfolio(returns, "max_sharpe", risk_free_rate=0.04)
    performance = portfolio_performance(
        optimized.values,
        returns.mean() * 252,
        returns.cov() * 252,
        risk_free_rate=0.04,
    )

    assert set(["Annual Return", "Annual Volatility", "Sharpe Ratio"]).issubset(summary)
    assert pd.notna(summary["Sharpe Ratio"])
    assert np.isclose(optimized.sum(), 1)
    assert all(np.isfinite(performance))
