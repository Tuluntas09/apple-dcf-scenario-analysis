<div align="center">

# 📊 Apple DCF Analysis — Macro-Driven Valuation

**Discounted Cash Flow valuation of Apple (AAPL) using real macroeconomic data from FRED API**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://python-apple-dcf-analysis-ocqwoknanfdkibaz7zdsyn.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![FRED API](https://img.shields.io/badge/Data-FRED%20API-1B2536?style=for-the-badge)](https://fred.stlouisfed.org)

</div>

---

## Overview

A Bloomberg-style financial dashboard that combines **macroeconomic data from the Federal Reserve (FRED API)** with **Apple's financial statements (Yahoo Finance)** to build a full DCF valuation model with scenario analysis. Built as a portfolio project targeting Financial Analyst and FP&A roles.

**Live:** https://python-apple-dcf-analysis-ocqwoknanfdkibaz7zdsyn.streamlit.app

---

## Key Findings

| Finding | Result |
|---------|--------|
| 📉 Fed rate +200bp impact | WACC +150bp → intrinsic value **−59%** |
| 🏆 Top value driver | FCF growth rate ($59/share impact range) |
| 🐂 Bull Case intrinsic value | **$223/share** |
| 📊 Base Case intrinsic value | **$148/share** |
| 🐻 Bear Case intrinsic value | **$92/share** |
| ⚡ Bull/Bear spread | **$131** (143% premium) |

---

## Features

| Feature | Description |
|---------|-------------|
| 📈 **DCF Model** | 5-year FCF projection · CAPM-based WACC · Gordon Growth terminal value |
| 🌍 **Scenario Analysis** | Bull / Base / Bear with macro-calibrated parameters |
| 🔥 **Sensitivity Heatmap** | WACC × Terminal Growth 2D grid — intrinsic value at every combination |
| 🌪️ **Tornado Chart** | Isolated impact of each variable on intrinsic value |
| 🃏 **KPI Cards** | Base value · Market price · WACC · Bull/Bear spread — live from FRED |
| ⚙️ **Interactive Sidebar** | FCF growth sliders · terminal growth · debt weight — instant recalculation |
| 📅 **Real-Time Data** | 10Y Treasury risk-free rate fetched live from FRED each session |
| 📓 **4 Jupyter Notebooks** | Full analysis pipeline: data collection → regression → DCF → scenarios |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data** | `yfinance` — Apple financials · `fredapi` — FRED macro series |
| **Processing** | `pandas` · `numpy` · `statsmodels` (OLS regression) |
| **Charts** | `plotly` — Bar, Pie, Heatmap, Scatter, Tornado |
| **UI** | `streamlit` — custom CSS, Bloomberg dark theme |
| **Notebooks** | `jupyter` — reproducible analysis pipeline |
| **Fonts** | Inter (Google Fonts) |
| **Deploy** | Streamlit Cloud (GitHub push-to-deploy) |

---

## Data Sources

| Indicator | Source | Series |
|-----------|--------|--------|
| Apple Revenue, FCF, Net Income | Yahoo Finance (`yfinance`) | AAPL |
| Fed Funds Rate | FRED | `FEDFUNDS` |
| CPI Inflation | FRED | `CPIAUCSL` |
| GDP Growth | FRED | `A191RL1Q225SBEA` |
| USD Index | FRED | `DTWEXBGS` |
| 10Y Treasury (Risk-Free Rate) | FRED | `DGS10` |

All FRED data freely available at [fred.stlouisfed.org](https://fred.stlouisfed.org).

---

## Architecture

```
Yahoo Finance (yfinance)      FRED API (fredapi)
         │                          │
         └──────────┬───────────────┘
                    ▼
           src/data_fetcher.py   ← API clients, raw fetch, fallback handling
                    │
                    ▼
           notebooks/01_data_collection.ipynb
           notebooks/02_macro_correlation_analysis.ipynb   ← OLS regression
                    │
                    ▼
           src/dcf_model.py      ← WACC · FCF projection · terminal value · sensitivity
                    │
           src/scenario_engine.py ← Bull / Base / Bear parameters · tornado inputs
                    │
                    ▼
           notebooks/03_dcf_model.ipynb
           notebooks/04_scenario_analysis.ipynb
                    │
                    ▼
           app/streamlit_app.py  ← Streamlit UI · Plotly charts · CSS · KPI cards
```

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/Tuluntas09/python-apple-dcf-analysis.git
cd python-apple-dcf-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
# Copy the secrets template:
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your FRED API key

# 4a. Run the Streamlit dashboard
streamlit run app/streamlit_app.py

# 4b. Or run the Jupyter notebooks
jupyter notebook
# Open notebooks in order: 01 → 02 → 03 → 04
```

> Get a free FRED API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

---

## Project Structure

```
python-apple-dcf-analysis/
├── app/
│   └── streamlit_app.py        # Streamlit dashboard (Bloomberg dark UI)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_macro_correlation_analysis.ipynb
│   ├── 03_dcf_model.ipynb
│   └── 04_scenario_analysis.ipynb
├── src/
│   ├── data_fetcher.py         # FRED + yfinance API clients
│   ├── dcf_model.py            # WACC, FCF projection, terminal value, sensitivity
│   └── scenario_engine.py     # Bull/Base/Bear scenario definitions + tornado
├── data/
│   └── processed/              # Combined CSV + exported charts
├── .streamlit/
│   └── secrets.toml.example    # API key template
└── requirements.txt
```

---

## Key Technical Decisions

- **Macro-calibrated FCF growth** — Base case growth rates derived from OLS regression of Apple revenue on GDP, Fed rate, CPI, USD index — not arbitrary assumptions
- **fredapi over pandas-datareader** — pandas-datareader 0.10.0 is broken with pandas 3.x; fredapi is actively maintained and directly supports FRED
- **Streamlit secrets for API keys** — hardcoded fallback removed from source; key stored in `.streamlit/secrets.toml` (gitignored) or Streamlit Cloud secrets
- **Plotly over matplotlib** — interactive tooltips, native dark theme, no white-background contrast issue in dark UI
- **`@st.cache_data(ttl=3600)`** — 1-hour cache avoids repeated Yahoo Finance / FRED calls; graceful fallback to FY2024 actuals on rate limit
- **Modular src/ layout** — `dcf_model.py` and `scenario_engine.py` are pure functions (no side effects), usable identically in notebooks and Streamlit app

---

## Related Projects

- [sql-financial-analysis](https://github.com/Tuluntas09/sql-financial-analysis) — PostgreSQL · 28 queries · 5-table e-commerce schema · financial metrics view
- [power-bi-financial-performance-dashboard](https://github.com/Tuluntas09/power-bi-financial-performance-dashboard) — Power BI · 21 DAX measures · 4-page dashboard
- [tcmb-macro-panel](https://github.com/Tuluntas09/tcmb-macro-panel) — Streamlit · TCMB EVDS API · live Turkish macro indicators

---

<div align="center">

*Built for Financial Analyst / FP&A portfolio · Data: FRED API + Yahoo Finance · Not financial advice*

**[Tuluntas09](https://github.com/Tuluntas09)**

</div>
