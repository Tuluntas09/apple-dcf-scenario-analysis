# Python Macro-Driven DCF Analysis — Apple (AAPL)

A financial analysis project combining macroeconomic data with fundamental valuation.
Built as part of a Financial Analyst / FP&A portfolio.

## What This Project Does

Uses real public data (FRED API + Yahoo Finance) to answer:

> *How do macroeconomic conditions — interest rates, GDP growth, inflation, and the USD — affect Apple's intrinsic value?*

**Key Findings:**
- Fed rate is the single most impactful variable: a 100bp rate hike reduces Apple's intrinsic value by ~10–15%
- GDP growth is the strongest revenue predictor (OLS regression: highest β coefficient, p < 0.05)
- Bull/Bear scenario gap spans ~$100+ per share — quantifying macro uncertainty on corporate valuation

## Project Structure

```
python-apple-dcf-analysis/
├── notebooks/
│   ├── 01_data_collection.ipynb          # Pull Apple financials + FRED macro data
│   ├── 02_macro_correlation_analysis.ipynb  # OLS regression + correlation heatmap
│   ├── 03_dcf_model.ipynb                # Base case DCF valuation
│   └── 04_scenario_analysis.ipynb        # Bull / Base / Bear + Tornado chart
├── data/
│   └── processed/                        # Charts and combined CSV
├── src/
│   ├── data_fetcher.py                   # FRED + yfinance utilities
│   ├── dcf_model.py                      # WACC, FCF projection, terminal value
│   └── scenario_engine.py               # Scenario definitions + comparison
└── requirements.txt
```

## Methodology

### Data Sources
| Data | Source | Series |
|------|--------|--------|
| Apple financials (Revenue, FCF) | Yahoo Finance (yfinance) | AAPL |
| Fed Funds Rate | FRED | FEDFUNDS |
| CPI Inflation | FRED | CPIAUCSL |
| GDP Growth | FRED | A191RL1Q225SBEA |
| USD Index | FRED | DTWEXBGS |
| 10Y Treasury (Risk-Free Rate) | FRED | DGS10 |

### DCF Model
- **WACC** = CAPM-based (Rf + β × MRP), blended with after-tax cost of debt
- **FCF Projection** = 5-year forward projection calibrated against historical growth + macro regression
- **Terminal Value** = Gordon Growth Model
- **Intrinsic Value** = PV(FCFs) + PV(TV) − Net Debt

### Scenarios
| Parameter | Bull | Base | Bear |
|-----------|------|------|------|
| GDP Growth | 3.5% | 2.5% | 1.0% |
| Fed Rate (Rf) | 3.5% | 4.5% | 5.5% |
| CPI | 2.0% | 3.0% | 4.5% |
| FCF Growth (Y1) | 12% | 8% | 3% |
| Terminal Growth | 3.5% | 3.0% | 2.0% |

## Setup & Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Open notebooks in order: `01 → 02 → 03 → 04`

## Related Projects

- [sql-financial-analysis](https://github.com/Tuluntas09/sql-financial-analysis) — PostgreSQL sales data analysis (28 queries, 5 tables)
- [power-bi-financial-performance-dashboard](https://github.com/Tuluntas09/power-bi-financial-performance-dashboard) — 4-page Power BI dashboard with 21 DAX measures

---
*Tools: Python, pandas, yfinance, pandas-datareader (FRED), statsmodels, matplotlib, seaborn*
