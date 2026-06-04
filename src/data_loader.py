from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.providers.yahoo import fetch_adjusted_close_prices


DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
SAMPLE_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv"


@dataclass(frozen=True)
class PriceDataResult:
    prices: pd.DataFrame
    source: str
    warning: str | None = None


def parse_tickers(raw_tickers: str) -> list[str]:
    tickers = [
        ticker.strip().upper()
        for ticker in raw_tickers.replace(";", ",").replace(" ", ",").split(",")
        if ticker.strip()
    ]
    return list(dict.fromkeys(tickers))


def fetch_yfinance_prices(
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    return fetch_adjusted_close_prices(tickers, start, end)


def load_sample_prices(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    prices = pd.read_csv(SAMPLE_DATA_PATH, parse_dates=["Date"]).set_index("Date")
    available = [ticker for ticker in tickers if ticker in prices.columns]
    if len(available) < 2:
        available = DEFAULT_TICKERS
    return prices[available].copy()


def load_prices(
    raw_tickers: str,
    start: date,
    end: date,
    data_mode: str = "sample",
) -> PriceDataResult:
    tickers = parse_tickers(raw_tickers) or DEFAULT_TICKERS

    if data_mode == "sample":
        prices = load_sample_prices(tickers)
        missing = sorted(set(tickers) - set(prices.columns))
        warning = None
        if missing:
            warning = f"Sample veri setinde bulunamadi: {', '.join(missing)}"
        return PriceDataResult(prices=prices, source="Sample veri", warning=warning)

    try:
        prices = fetch_yfinance_prices(tickers, start, end)
        missing = sorted(set(tickers) - set(prices.columns))
        warning = None
        if missing:
            warning = f"Kullanilabilir kapanis fiyati bulunamadi: {', '.join(missing)}"
        return PriceDataResult(prices=prices, source="Yahoo Finance", warning=warning)
    except Exception as exc:
        return PriceDataResult(
            prices=pd.DataFrame(),
            source="Yahoo Finance",
            warning=(
                "Canli veri yuklenemedi. Baglanti, sembol veya tarih araligini kontrol et. "
                f"Neden: {exc}"
            ),
        )
