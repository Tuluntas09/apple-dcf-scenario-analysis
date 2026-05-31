"""
Data fetching utilities: Apple financials via yfinance, macro indicators via FRED.
"""

import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime


# ---------------------------------------------------------------------------
# FRED API key: Streamlit secrets > env var > hardcoded fallback
import os as _os
try:
    import streamlit as _st
    FRED_API_KEY = _st.secrets["FRED_API_KEY"]
except Exception:
    FRED_API_KEY = _os.environ.get("FRED_API_KEY", "")
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "fed_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "gdp_growth": "A191RL1Q225SBEA",
    "usd_index": "DTWEXBGS",
    "treasury_10y": "DGS10",
}

START_DATE = "2013-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def fetch_apple_financials() -> pd.DataFrame:
    """Return annual Apple income statement + cash flow merged on fiscal year."""
    ticker = yf.Ticker("AAPL")

    income = ticker.financials.T
    cashflow = ticker.cashflow.T

    income.index = pd.to_datetime(income.index)
    cashflow.index = pd.to_datetime(cashflow.index)

    cols_income = ["Total Revenue", "Gross Profit", "Net Income"]
    cols_cf = ["Free Cash Flow", "Capital Expenditure", "Operating Cash Flow"]

    income = income[[c for c in cols_income if c in income.columns]]
    cashflow = cashflow[[c for c in cols_cf if c in cashflow.columns]]

    df = income.join(cashflow, how="outer").sort_index()
    df.index.name = "date"
    return df


def fetch_fred_series(series_id: str, freq: str = "A") -> pd.Series:
    """Fetch a FRED series via fredapi and resample to annual mean."""
    fred = Fred(api_key=FRED_API_KEY)
    raw = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
    raw.index = pd.to_datetime(raw.index)
    annual = raw.resample(freq).mean()
    annual.name = series_id
    return annual


def fetch_all_macro() -> pd.DataFrame:
    """Fetch all macro indicators from FRED, return annual DataFrame."""
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            frames[name] = fetch_fred_series(series_id)
        except Exception as e:
            print(f"Warning: could not fetch {series_id} — {e}")
    return pd.DataFrame(frames)


def build_combined_dataset(save_path: str = None) -> pd.DataFrame:
    """
    Merge Apple annual financials with macro indicators.
    Aligns on fiscal year (Apple's FY ends in September).
    """
    apple = fetch_apple_financials()
    macro = fetch_all_macro()

    apple.index = apple.index.year
    macro.index = macro.index.year

    df = apple.join(macro, how="inner")
    df.index.name = "fiscal_year"

    if save_path:
        df.to_csv(save_path)
        print(f"Saved to {save_path}")

    return df


def fetch_apple_beta() -> float:
    """Return Apple's 5-year monthly beta from yfinance."""
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    return info.get("beta", 1.24)


def fetch_shares_outstanding() -> float:
    """Return Apple's shares outstanding (in billions)."""
    ticker = yf.Ticker("AAPL")
    info = ticker.info
    return info.get("sharesOutstanding", 15_500_000_000)
