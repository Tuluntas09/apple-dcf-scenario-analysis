import numpy as np
import pandas as pd

from src.advanced_analytics import (
    rebalancing_comparison,
    rolling_metrics,
    simulate_rebalanced_returns,
    stress_test_summary,
    transaction_cost_impact,
)


def sample_returns():
    index = pd.date_range("2024-01-02", periods=80, freq="W-TUE")
    return pd.DataFrame(
        {
            "AAA": np.linspace(0.001, 0.01, 80),
            "BBB": np.linspace(0.002, -0.002, 80),
            "CCC": np.sin(np.linspace(0, 6, 80)) / 100,
        },
        index=index,
    )


def test_rebalanced_returns_are_non_empty():
    returns = sample_returns()
    result = simulate_rebalanced_returns(returns, [0.4, 0.4, 0.2], "Aylik")

    assert not result.empty
    assert result.index.is_monotonic_increasing


def test_rebalancing_comparison_has_all_frequencies():
    comparison = rebalancing_comparison(sample_returns(), [0.4, 0.4, 0.2])

    assert {"Aylik", "Ceyreklik", "Yillik"}.issubset(comparison.index)


def test_rolling_metrics_outputs_expected_columns():
    returns = pd.Series(np.linspace(-0.01, 0.01, 40), index=pd.date_range("2024-01-01", periods=40))
    result = rolling_metrics(returns, window=10)

    assert {"Rolling Return", "Rolling Volatility", "Rolling Sharpe"}.issubset(result.columns)


def test_stress_test_summary_returns_named_scenarios():
    returns = pd.Series(0.001, index=pd.date_range("2024-04-01", periods=60))
    result = stress_test_summary(returns)

    assert "2024 Bahar Geri Cekilme" in result.index


def test_transaction_cost_impact_uses_turnover():
    current = pd.Series({"AAA": 0.5, "BBB": 0.5})
    target = pd.Series({"AAA": 0.7, "BBB": 0.3})
    result = transaction_cost_impact(current, target, cost_bps=10)

    assert np.isclose(result["Turnover"], 0.4)
    assert np.isclose(result["Tahmini Maliyet"], 0.0004)
