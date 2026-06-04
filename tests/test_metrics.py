import numpy as np
import pandas as pd

from src.metrics import (
    benchmark_comparison,
    calculate_returns,
    concentration_summary,
    effective_number_of_assets,
    max_drawdown,
    normalize_weights,
    portfolio_returns,
    risk_contribution,
    sharpe_ratio,
)


def test_calculate_returns_drops_first_row():
    prices = pd.DataFrame({"AAA": [100, 110, 121], "BBB": [50, 55, 60.5]})
    returns = calculate_returns(prices)

    assert len(returns) == 2
    assert np.isclose(returns["AAA"].iloc[0], 0.10)
    assert np.isclose(returns["BBB"].iloc[1], 0.10)


def test_weights_are_normalized():
    weights = normalize_weights([20, 30, 50], 3)

    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[0], 0.2)


def test_portfolio_returns_uses_weights():
    returns = pd.DataFrame({"AAA": [0.10, 0.00], "BBB": [0.00, 0.10]})
    result = portfolio_returns(returns, [0.25, 0.75])

    assert np.isclose(result.iloc[0], 0.025)
    assert np.isclose(result.iloc[1], 0.075)


def test_max_drawdown_detects_peak_to_trough_loss():
    returns = pd.Series([0.10, -0.20, 0.05])

    assert np.isclose(max_drawdown(returns), -0.20)


def test_sharpe_ratio_is_finite_for_variable_returns():
    returns = pd.Series([0.01, -0.005, 0.007, 0.002, -0.003])

    assert np.isfinite(sharpe_ratio(returns))


def test_effective_number_of_assets_reflects_concentration():
    assert np.isclose(effective_number_of_assets([0.5, 0.5]), 2.0)
    assert effective_number_of_assets([0.9, 0.1]) < 2.0


def test_concentration_summary_contains_top_asset():
    summary = concentration_summary([0.7, 0.3], ["AAA", "BBB"])

    assert summary.loc[0, "Deger"] == "AAA"


def test_risk_contribution_sums_to_one():
    returns = pd.DataFrame({"AAA": [0.01, 0.02, -0.01], "BBB": [0.002, 0.001, 0.003]})
    contribution = risk_contribution(returns, [0.5, 0.5])

    assert np.isclose(contribution["Risk Contribution"].sum(), 1.0)


def test_benchmark_comparison_has_difference_row():
    portfolio = pd.Series([0.01, 0.02, -0.01, 0.005])
    benchmark = pd.Series([0.008, 0.015, -0.008, 0.003])
    comparison = benchmark_comparison(portfolio, benchmark)

    assert "Fark" in comparison.index
    assert "Tracking Error" in comparison.columns
