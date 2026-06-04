import numpy as np
import pandas as pd

from src.optimization import optimize_portfolio, portfolio_performance, random_portfolios
from src.simulation import monte_carlo_paths, terminal_value_summary


def sample_returns():
    return pd.DataFrame(
        {
            "AAA": [0.01, 0.02, -0.01, 0.005, 0.004],
            "BBB": [0.002, 0.001, 0.003, 0.002, 0.001],
            "CCC": [-0.004, 0.006, 0.005, -0.002, 0.003],
        }
    )


def test_optimize_portfolio_returns_valid_weights():
    weights = optimize_portfolio(sample_returns(), "max_sharpe", risk_free_rate=0.0)

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_random_portfolios_have_expected_columns():
    portfolios = random_portfolios(sample_returns(), n_portfolios=10, seed=1)

    assert len(portfolios) == 10
    assert {"Annual Return", "Annual Volatility", "Sharpe", "AAA", "BBB", "CCC"}.issubset(portfolios.columns)


def test_portfolio_performance_outputs_finite_values():
    returns = sample_returns()
    expected_returns = returns.mean() * 252
    covariance = returns.cov() * 252

    annual_return, volatility, sharpe = portfolio_performance(
        np.array([0.4, 0.4, 0.2]),
        expected_returns,
        covariance,
    )

    assert np.isfinite(annual_return)
    assert volatility >= 0
    assert np.isfinite(sharpe)


def test_monte_carlo_summary_contains_ordered_scenarios():
    returns = pd.Series([0.01, -0.005, 0.004, 0.002, -0.001])
    paths = monte_carlo_paths(returns, initial_value=1000, horizon_days=30, n_simulations=50, seed=2)
    summary = terminal_value_summary(paths)

    assert paths.shape == (30, 50)
    assert summary["P05"] <= summary["Median"] <= summary["P95"]
