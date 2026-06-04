from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = prices.copy()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(axis=1, how="all").ffill().dropna()
    return cleaned.loc[:, cleaned.nunique(dropna=True) > 1]


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_prices(prices)
    return cleaned.pct_change(fill_method=None).dropna(how="all")


def normalize_weights(weights: np.ndarray | list[float], n_assets: int) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.size != n_assets:
        raise ValueError("Weights length must match the number of assets.")
    if np.isclose(weights_array.sum(), 0):
        return np.repeat(1 / n_assets, n_assets)
    return weights_array / weights_array.sum()


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray | list[float]) -> pd.Series:
    normalized = normalize_weights(weights, returns.shape[1])
    return returns.dot(normalized).rename("Portfolio")


def cumulative_growth(returns: pd.Series | pd.DataFrame, initial_value: float = 1.0):
    return initial_value * (1 + returns).cumprod()


def annualized_return(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    if len(returns) == 0:
        return np.nan
    compounded = (1 + returns).prod()
    result = compounded ** (TRADING_DAYS / len(returns)) - 1
    if isinstance(result, pd.Series):
        return result
    return float(result)


def annualized_volatility(returns: pd.Series | pd.DataFrame) -> pd.Series | float:
    result = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    if isinstance(result, pd.Series):
        return result
    return float(result)


def downside_volatility(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    daily_target = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    downside = returns[returns < daily_target] - daily_target
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess_return = annualized_return(returns) - risk_free_rate
    volatility = annualized_volatility(returns)
    if np.isclose(volatility, 0) or np.isnan(volatility):
        return np.nan
    return float(excess_return / volatility)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess_return = annualized_return(returns) - risk_free_rate
    downside = downside_volatility(returns, risk_free_rate)
    if np.isclose(downside, 0) or np.isnan(downside):
        return np.nan
    return float(excess_return / downside)


def max_drawdown(returns: pd.Series) -> float:
    growth = cumulative_growth(returns)
    drawdown = growth / growth.cummax() - 1
    return float(drawdown.min())


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    return float(np.quantile(returns.dropna(), 1 - confidence))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if not tail.empty else var


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan
    covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    variance = aligned.iloc[:, 1].var()
    if np.isclose(variance, 0):
        return np.nan
    return float(covariance / variance)


def asset_metric_table(
    returns: pd.DataFrame,
    benchmark: str | None = None,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    benchmark_returns = returns[benchmark] if benchmark in returns.columns else None
    rows = []
    for column in returns.columns:
        series = returns[column].dropna()
        rows.append(
            {
                "Asset": column,
                "Annual Return": annualized_return(series),
                "Annual Volatility": annualized_volatility(series),
                "Sharpe": sharpe_ratio(series, risk_free_rate),
                "Sortino": sortino_ratio(series, risk_free_rate),
                "Max Drawdown": max_drawdown(series),
                "Beta": beta(series, benchmark_returns) if benchmark_returns is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("Asset")


def portfolio_metric_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, float]:
    return {
        "Annual Return": annualized_return(returns),
        "Annual Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate),
        "Max Drawdown": max_drawdown(returns),
        f"VaR {int(confidence * 100)}%": value_at_risk(returns, confidence),
        f"CVaR {int(confidence * 100)}%": conditional_value_at_risk(returns, confidence),
    }


def effective_number_of_assets(weights: np.ndarray | list[float]) -> float:
    weights_array = np.asarray(weights, dtype=float)
    weights_array = weights_array / weights_array.sum()
    return float(1 / np.sum(weights_array**2))


def concentration_summary(weights: np.ndarray | list[float], assets: list[str]) -> pd.DataFrame:
    weights_array = np.asarray(weights, dtype=float)
    weights_array = weights_array / weights_array.sum()
    sorted_pairs = sorted(zip(assets, weights_array), key=lambda item: item[1], reverse=True)
    top_asset, top_weight = sorted_pairs[0]
    return pd.DataFrame(
        [
            ("En buyuk pozisyon", top_asset),
            ("En buyuk agirlik", top_weight),
            ("Efektif varlik sayisi", effective_number_of_assets(weights_array)),
        ],
        columns=["Metrik", "Deger"],
    )


def benchmark_comparison(
    portfolio: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    aligned = pd.concat([portfolio.rename("Portfoy"), benchmark.rename("Benchmark")], axis=1).dropna()
    excess = aligned["Portfoy"] - aligned["Benchmark"]
    tracking_error = excess.std(ddof=1) * np.sqrt(TRADING_DAYS)
    information_ratio = (
        annualized_return(excess) / tracking_error
        if not np.isclose(tracking_error, 0)
        else np.nan
    )
    rows = {
        "Portfoy": {
            "Yillik Getiri": annualized_return(aligned["Portfoy"]),
            "Yillik Risk": annualized_volatility(aligned["Portfoy"]),
            "Sharpe": sharpe_ratio(aligned["Portfoy"], risk_free_rate),
            "Max Dusus": max_drawdown(aligned["Portfoy"]),
        },
        "Benchmark": {
            "Yillik Getiri": annualized_return(aligned["Benchmark"]),
            "Yillik Risk": annualized_volatility(aligned["Benchmark"]),
            "Sharpe": sharpe_ratio(aligned["Benchmark"], risk_free_rate),
            "Max Dusus": max_drawdown(aligned["Benchmark"]),
        },
        "Fark": {
            "Yillik Getiri": annualized_return(aligned["Portfoy"]) - annualized_return(aligned["Benchmark"]),
            "Yillik Risk": annualized_volatility(aligned["Portfoy"]) - annualized_volatility(aligned["Benchmark"]),
            "Sharpe": sharpe_ratio(aligned["Portfoy"], risk_free_rate) - sharpe_ratio(aligned["Benchmark"], risk_free_rate),
            "Max Dusus": max_drawdown(aligned["Portfoy"]) - max_drawdown(aligned["Benchmark"]),
        },
    }
    comparison = pd.DataFrame(rows).T
    comparison["Tracking Error"] = np.nan
    comparison["Information Ratio"] = np.nan
    comparison.loc["Fark", "Tracking Error"] = tracking_error
    comparison.loc["Fark", "Information Ratio"] = information_ratio
    return comparison


def risk_contribution(returns: pd.DataFrame, weights: np.ndarray | list[float]) -> pd.DataFrame:
    weights_array = normalize_weights(weights, returns.shape[1])
    covariance = returns.cov() * TRADING_DAYS
    portfolio_variance = float(weights_array.T @ covariance.values @ weights_array)
    if np.isclose(portfolio_variance, 0):
        contributions = np.zeros_like(weights_array)
    else:
        marginal_contribution = covariance.values @ weights_array
        contributions = weights_array * marginal_contribution / portfolio_variance
    return pd.DataFrame(
        {
            "Asset": returns.columns,
            "Weight": weights_array,
            "Risk Contribution": contributions,
        }
    ).set_index("Asset")
