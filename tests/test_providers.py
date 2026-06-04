import os

from src.providers.finnhub import fetch_finnhub_snapshot, has_finnhub_key
from src.secrets import get_secret


def test_get_secret_reads_environment(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

    assert get_secret("FINNHUB_API_KEY") == "test-key"


def test_finnhub_key_detection_without_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr("src.providers.finnhub.get_secret", lambda name: None)

    assert has_finnhub_key() is False


def test_finnhub_snapshot_without_key_is_graceful(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr("src.providers.finnhub.get_secret", lambda name: None)

    snapshot = fetch_finnhub_snapshot(["AAPL"])

    assert snapshot.profiles == []
    assert snapshot.news == []
    assert "API key" in snapshot.warning
