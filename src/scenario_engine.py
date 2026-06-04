"""
Scenario engine: Bull / Base / Bear case parameter sets and comparison utilities.
"""

import pandas as pd


SCENARIOS = {
    "Bull": {
        "label": "Bull Case",
        "color": "#2ecc71",
        "gdp_growth": 0.035,
        "fed_rate": 0.035,
        "cpi": 0.020,
        "usd": "weak",
        "fcf_growth_rates": [0.12, 0.12, 0.11, 0.10, 0.09],
        "terminal_growth": 0.035,
        "risk_free_rate": 0.035,
        "description": "Low rates, strong GDP, weak USD boost international revenue.",
    },
    "Base": {
        "label": "Base Case",
        "color": "#3498db",
        "gdp_growth": 0.025,
        "fed_rate": 0.045,
        "cpi": 0.030,
        "usd": "neutral",
        "fcf_growth_rates": [0.08, 0.08, 0.07, 0.07, 0.06],
        "terminal_growth": 0.030,
        "risk_free_rate": 0.045,
        "description": "Current macro environment, moderate growth continuing.",
    },
    "Bear": {
        "label": "Bear Case",
        "color": "#e74c3c",
        "gdp_growth": 0.010,
        "fed_rate": 0.055,
        "cpi": 0.045,
        "usd": "strong",
        "fcf_growth_rates": [0.03, 0.03, 0.02, 0.02, 0.01],
        "terminal_growth": 0.020,
        "risk_free_rate": 0.055,
        "description": "High rates, low GDP, strong USD compressing international margins.",
    },
}


def get_scenario(name: str) -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def build_scenario_summary(dcf_results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing key metrics across scenarios.
    dcf_results: {scenario_name: dcf_model.intrinsic_value_per_share() output}
    """
    rows = []
    for name, result in dcf_results.items():
        scenario = SCENARIOS[name]
        rows.append({
            "Scenario": scenario["label"],
            "FCF Büyümesi (Y1)": f"{scenario['fcf_growth_rates'][0]:.0%}",
            "Terminal Growth": f"{scenario['terminal_growth']:.1%}",
            "Risk-Free Rate": f"{scenario['risk_free_rate']:.1%}",
            "WACC": f"{result['wacc']:.2%}",
            "Enterprise Value ($B)": round(result["enterprise_value"] / 1e9, 1),
            "Intrinsic Value / Share ($)": round(result["intrinsic_value_per_share"], 2),
        })
    return pd.DataFrame(rows).set_index("Scenario")


def tornado_inputs(
    base_fcf: float,
    base_result: dict,
    net_debt: float,
    shares_outstanding: float,
    beta: float,
) -> pd.DataFrame:
    """
    Tornado chart data: vary each input ±1 standard deviation, measure value impact.
    Returns DataFrame with variable name, low value, high value, and delta.
    """
    from src.dcf_model import calculate_wacc, intrinsic_value_per_share

    base_val = base_result["intrinsic_value_per_share"]
    base_growth = [0.08, 0.08, 0.07, 0.07, 0.06]
    base_rf = 0.045
    base_tg = 0.030

    def run(rf=base_rf, growth_rates=None, tg=base_tg, b=beta):
        gr = growth_rates if growth_rates else base_growth
        w = calculate_wacc(rf, b)
        r = intrinsic_value_per_share(base_fcf, gr, w, tg, net_debt, shares_outstanding)
        return r["intrinsic_value_per_share"]

    rows = [
        {
            "Variable": "Fed Faizi (Risk-Free Rate)",
            "Low ($)": run(rf=0.035),
            "Base ($)": base_val,
            "High ($)": run(rf=0.055),
        },
        {
            "Variable": "FCF Büyüme Oranı",
            "Low ($)": run(growth_rates=[0.03] * 5),
            "Base ($)": base_val,
            "High ($)": run(growth_rates=[0.12] * 5),
        },
        {
            "Variable": "Terminal Growth Rate",
            "Low ($)": run(tg=0.02),
            "Base ($)": base_val,
            "High ($)": run(tg=0.035),
        },
        {
            "Variable": "Beta",
            "Low ($)": run(b=beta * 0.8),
            "Base ($)": base_val,
            "High ($)": run(b=beta * 1.2),
        },
    ]

    df = pd.DataFrame(rows)
    df["Delta ($)"] = df["High ($)"] - df["Low ($)"]
    return df.sort_values("Delta ($)", ascending=False)
