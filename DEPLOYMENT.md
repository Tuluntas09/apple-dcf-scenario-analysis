# Deployment Notes

## Local Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Create a new Streamlit app from the repository.
3. Set the main file path to `app.py`.
4. Use Python 3.11 or newer.
5. Yahoo Finance price data is used by default; optional Finnhub enrichment requires a secret.

## Render

Use a web service with this start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Data Modes

- `Sample veri`: bundled offline dataset, best for demos and CI-safe use.
- `Canli veri`: Yahoo Finance via `yfinance`; requires network access.
- `Finnhub sirket verisi`: optional company profile/news enrichment; requires `FINNHUB_API_KEY`.

## Finnhub Secret

For local development, create:

```toml
# .streamlit/secrets.toml
FINNHUB_API_KEY = "your_finnhub_api_key"
```

For Streamlit Community Cloud, add the same value in the app secrets panel.

For Render, add `FINNHUB_API_KEY` as an environment variable.

## CI

The GitHub Actions workflow in `.github/workflows/tests.yml` installs dependencies and runs:

```bash
python -m pytest
```
