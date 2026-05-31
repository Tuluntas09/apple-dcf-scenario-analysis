"""
DCF valuation engine: WACC, FCF projection, terminal value, intrinsic value.
"""

import numpy as np
import pandas as pd


MARKET_RISK_PREMIUM = 0.055  # Damodaran long-run US equity risk premium


def calculate_wacc(
    risk_free_rate: float,
    beta: float,
    market_risk_premium: float = MARKET_RISK_PREMIUM,
    debt_weight: float = 0.25,
    cost_of_debt: float = 0.035,
    tax_rate: float = 0.21,
) -> float:
    """
    WACC = E/V * (Rf + β * MRP) + D/V * Kd * (1 - T)
    Apple's capital structure: ~75% equity, ~25% debt (approximate).
    """
    equity_weight = 1 - debt_weight
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    wacc = (equity_weight * cost_of_equity) + (
        debt_weight * cost_of_debt * (1 - tax_rate)
    )
    return wacc


def project_fcf(
    base_fcf: float,
    growth_rates: list[float],
) -> list[float]:
    """
    Project free cash flows for each year given a list of annual growth rates.
    growth_rates: e.g. [0.08, 0.08, 0.07, 0.07, 0.06] for 5 years
    """
    projected = []
    fcf = base_fcf
    for g in growth_rates:
        fcf = fcf * (1 + g)
        projected.append(fcf)
    return projected


def terminal_value(final_year_fcf: float, wacc: float, terminal_growth: float = 0.03) -> float:
    """Gordon Growth Model terminal value at end of projection period."""
    if wacc <= terminal_growth:
        raise ValueError("WACC must exceed terminal growth rate.")
    return final_year_fcf * (1 + terminal_growth) / (wacc - terminal_growth)


def intrinsic_value_per_share(
    base_fcf: float,
    growth_rates: list[float],
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
) -> dict:
    """
    Full DCF: project FCFs → discount → terminal value → equity value → per share.

    Returns a dict with intermediate results for transparency.
    """
    fcfs = project_fcf(base_fcf, growth_rates)

    pv_fcfs = [fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(fcfs)]

    tv = terminal_value(fcfs[-1], wacc, terminal_growth)
    pv_tv = tv / (1 + wacc) ** len(fcfs)

    enterprise_value = sum(pv_fcfs) + pv_tv
    equity_value = enterprise_value - net_debt
    price_per_share = equity_value / shares_outstanding

    return {
        "projected_fcfs": fcfs,
        "pv_fcfs": pv_fcfs,
        "sum_pv_fcfs": sum(pv_fcfs),
        "terminal_value": tv,
        "pv_terminal_value": pv_tv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_value_per_share": price_per_share,
        "wacc": wacc,
        "terminal_growth": terminal_growth,
    }


def sensitivity_table(
    base_fcf: float,
    growth_rates: list[float],
    net_debt: float,
    shares_outstanding: float,
    beta: float,
    wacc_range: list[float] = None,
    tg_range: list[float] = None,
) -> pd.DataFrame:
    """
    2D sensitivity: WACC (rows) × Terminal Growth Rate (columns) → intrinsic value per share.
    """
    if wacc_range is None:
        wacc_range = [round(w, 3) for w in np.arange(0.07, 0.14, 0.01)]
    if tg_range is None:
        tg_range = [round(g, 3) for g in np.arange(0.01, 0.06, 0.01)]

    rows = {}
    for wacc in wacc_range:
        row = {}
        for tg in tg_range:
            result = intrinsic_value_per_share(
                base_fcf, growth_rates, wacc, tg, net_debt, shares_outstanding
            )
            row[f"{tg:.0%}"] = round(result["intrinsic_value_per_share"], 2)
        rows[f"{wacc:.1%}"] = row

    df = pd.DataFrame(rows).T
    df.index.name = "WACC"
    df.columns.name = "Terminal Growth"
    return df
