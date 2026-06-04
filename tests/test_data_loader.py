from datetime import date

from src.data_loader import load_prices, load_sample_prices, parse_tickers


def test_parse_tickers_normalizes_and_deduplicates():
    assert parse_tickers("spy, qqq; SPY tlt") == ["SPY", "QQQ", "TLT"]


def test_sample_prices_load_offline_data():
    prices = load_sample_prices(["SPY", "QQQ"])

    assert list(prices.columns) == ["SPY", "QQQ"]
    assert len(prices) >= 50


def test_live_prices_returns_clear_error_without_sample_fallback(monkeypatch):
    def raise_error(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("src.data_loader.fetch_yfinance_prices", raise_error)

    result = load_prices("SPY, QQQ", date(2024, 1, 1), date(2024, 12, 31), data_mode="live")

    assert result.source == "Yahoo Finance"
    assert result.prices.empty
    assert "Canli veri yuklenemedi" in result.warning
