from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
import json

from src.secrets import get_secret


FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


@dataclass(frozen=True)
class FinnhubCompany:
    ticker: str
    name: str | None
    exchange: str | None
    industry: str | None
    country: str | None
    market_cap: float | None


@dataclass(frozen=True)
class FinnhubNewsItem:
    ticker: str
    headline: str
    source: str | None
    url: str | None
    published_at: str | None


@dataclass(frozen=True)
class FinnhubSnapshot:
    profiles: list[FinnhubCompany]
    news: list[FinnhubNewsItem]
    warning: str | None = None


def has_finnhub_key() -> bool:
    return bool(get_secret("FINNHUB_API_KEY"))


def _request_json(path: str, params: dict[str, str]) -> dict | list:
    api_key = get_secret("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY bulunamadi.")
    query = urlencode({**params, "token": api_key})
    url = f"{FINNHUB_BASE_URL}/{path}?{query}"
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_company_profile(ticker: str) -> FinnhubCompany:
    payload = _request_json("stock/profile2", {"symbol": ticker})
    if not isinstance(payload, dict) or not payload:
        return FinnhubCompany(ticker, None, None, None, None, None)
    return FinnhubCompany(
        ticker=ticker,
        name=payload.get("name"),
        exchange=payload.get("exchange"),
        industry=payload.get("finnhubIndustry"),
        country=payload.get("country"),
        market_cap=payload.get("marketCapitalization"),
    )


def fetch_company_news(ticker: str, limit: int = 3) -> list[FinnhubNewsItem]:
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=14)
    payload = _request_json(
        "company-news",
        {"symbol": ticker, "from": start.isoformat(), "to": today.isoformat()},
    )
    if not isinstance(payload, list):
        return []
    items = []
    for row in payload[:limit]:
        published_at = None
        if row.get("datetime"):
            published_at = datetime.fromtimestamp(row["datetime"], tz=timezone.utc).date().isoformat()
        items.append(
            FinnhubNewsItem(
                ticker=ticker,
                headline=row.get("headline", ""),
                source=row.get("source"),
                url=row.get("url"),
                published_at=published_at,
            )
        )
    return items


def fetch_finnhub_snapshot(tickers: list[str], news_per_ticker: int = 2) -> FinnhubSnapshot:
    if not has_finnhub_key():
        return FinnhubSnapshot([], [], "Finnhub API key eklenmemis.")
    profiles: list[FinnhubCompany] = []
    news: list[FinnhubNewsItem] = []
    warnings: list[str] = []
    for ticker in tickers:
        try:
            profiles.append(fetch_company_profile(ticker))
            news.extend(fetch_company_news(ticker, limit=news_per_ticker))
        except Exception as exc:
            warnings.append(f"{ticker}: {exc}")
    return FinnhubSnapshot(profiles, news, "; ".join(warnings) if warnings else None)
