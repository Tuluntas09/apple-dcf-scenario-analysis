# Apple DCF Scenario Analysis

Macro-driven Apple valuation dashboard using FRED data, Yahoo Finance
fundamentals, DCF scenario analysis, sensitivity tables, and Streamlit.

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://python-apple-dcf-analysis-ocqwoknanfdkibaz7zdsyn.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![FRED](https://img.shields.io/badge/Data-FRED%20API-1B2536?style=for-the-badge)](https://fred.stlouisfed.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

## 30-Second Scan

| Area | What this project shows |
|---|---|
| Finance workflow | End-to-end DCF valuation case study for Apple with macro-calibrated assumptions |
| Modeling | CAPM-based WACC, 5-year FCF projection, Gordon Growth terminal value, bull/base/bear scenarios |
| Sensitivity | WACC vs terminal growth heatmap and tornado chart for value-driver analysis |
| Data | FRED macro series and Yahoo Finance financial statement data |
| Portfolio value | A recruiter-friendly financial modeling project with interactive dashboard and reproducible notebooks |

> Educational valuation analysis only. This project does not provide investment
> advice, trading signals, or a recommendation to buy or sell securities.

## Streamlit App

Open the Streamlit deployment link:
[python-apple-dcf-analysis-ocqwoknanfdkibaz7zdsyn.streamlit.app](https://python-apple-dcf-analysis-ocqwoknanfdkibaz7zdsyn.streamlit.app)

## Key Outputs

| Output | Purpose |
|---|---|
| DCF model | Computes intrinsic value from free cash flow, WACC, and terminal value assumptions |
| Scenario analysis | Compares bull, base, and bear valuation cases |
| Sensitivity heatmap | Shows how value changes across WACC and terminal growth assumptions |
| Tornado chart | Ranks the most important drivers of valuation range |
| Notebooks | Reproducible data collection, regression, DCF, and scenario workflow |

## Tech Stack

| Layer | Tools |
|---|---|
| App | Python, Streamlit |
| Data | `yfinance`, `fredapi` |
| Analytics | pandas, numpy, statsmodels |
| Charts | Plotly |
| Notebooks | Jupyter |
| Deployment | Streamlit Cloud |

## Data Sources

| Data | Source |
|---|---|
| Apple financials | Yahoo Finance via `yfinance` |
| Fed Funds Rate | FRED `FEDFUNDS` |
| CPI inflation | FRED `CPIAUCSL` |
| GDP growth | FRED `A191RL1Q225SBEA` |
| USD index | FRED `DTWEXBGS` |
| 10Y Treasury rate | FRED `DGS10` |

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project Structure

```text
app/        Streamlit application
src/        Data, modeling, and charting modules
notebooks/  Reproducible analysis workflow
data/       Local input/output data files
```

## Validation

- Run the Streamlit app locally and confirm the dashboard loads.
- Open notebooks in order to review the analysis pipeline.
- Confirm the Streamlit app link, FRED links, and Yahoo/FRED data notes render correctly on GitHub.

## License

MIT License. See [LICENSE](LICENSE).
