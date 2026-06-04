from __future__ import annotations

from datetime import date

import pandas as pd


def fetch_adjusted_close_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
        timeout=20,
    )

    if data.empty:
        raise ValueError("Piyasa verisi donmedi.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Indirilen veri kapanis fiyatlarini icermiyor.")
        prices = data["Close"]
    else:
        if "Close" not in data:
            raise ValueError("Indirilen veri kapanis fiyatlarini icermiyor.")
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index().dropna(axis=1, how="all").ffill().dropna()
    prices = prices.loc[:, ~prices.columns.duplicated()]

    if prices.empty:
        raise ValueError("Temizleme sonrasi kapanis fiyatlari bos kaldi.")

    return prices
