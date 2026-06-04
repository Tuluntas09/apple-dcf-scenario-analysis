"""
Apple DCF Analysis — Interactive Streamlit Dashboard
Portfolio project: Financial Analyst / FP&A
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from fredapi import Fred

from src.dcf_model import (
    calculate_wacc, intrinsic_value_per_share,
    sensitivity_table, MARKET_RISK_PREMIUM
)
from src.scenario_engine import SCENARIOS, build_scenario_summary, tornado_inputs

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL · DCF Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG        = "#0a0e1a"
BG_CARD   = "#111827"
BG_CARD2  = "#1a2235"
BORDER    = "#1e2d45"
ACCENT    = "#3b82f6"      # electric blue
GREEN     = "#10b981"
RED       = "#ef4444"
YELLOW    = "#f59e0b"
TEXT      = "#e2e8f0"
TEXT_DIM  = "#64748b"
FONT      = "'Inter', 'Segoe UI', sans-serif"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT, size=12),
)

# Reusable axis styles
XAXIS = dict(showgrid=False, zeroline=False, color=TEXT_DIM,
             linecolor=BORDER, tickfont=dict(color=TEXT_DIM))
YAXIS = dict(showgrid=True, gridcolor=BORDER, zeroline=False,
             color=TEXT_DIM, linecolor=BORDER, tickfont=dict(color=TEXT_DIM))

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: {FONT};
    background-color: {BG};
    color: {TEXT};
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background-color: {BG_CARD} !important;
    border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span {{
    color: {TEXT_DIM} !important;
    font-size: 11px !important;
  }}

  /* Metric cards */
  .kpi-grid {{ display: flex; gap: 16px; margin-bottom: 24px; }}
  .kpi-card {{
    flex: 1;
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
  }}
  .kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {ACCENT}, {GREEN});
  }}
  .kpi-label {{
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin-bottom: 8px;
  }}
  .kpi-value {{
    font-size: 28px;
    font-weight: 700;
    color: {TEXT};
    line-height: 1;
  }}
  .kpi-sub {{
    font-size: 12px;
    margin-top: 6px;
  }}
  .kpi-up   {{ color: {GREEN}; }}
  .kpi-down {{ color: {RED};   }}
  .kpi-neu  {{ color: {TEXT_DIM}; }}

  /* Section headers */
  .section-header {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin: 24px 0 16px;
  }}

  /* Page title */
  .page-title {{
    font-size: 26px;
    font-weight: 700;
    color: {TEXT};
    letter-spacing: -.02em;
  }}
  .page-subtitle {{
    font-size: 13px;
    color: {TEXT_DIM};
    margin-top: 4px;
  }}
  .ticker-badge {{
    display: inline-block;
    background: {ACCENT}22;
    border: 1px solid {ACCENT}55;
    color: {ACCENT};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .08em;
    padding: 2px 10px;
    border-radius: 20px;
    margin-left: 10px;
    vertical-align: middle;
  }}

  /* Table */
  .styled-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .styled-table th {{
    background: {BG_CARD2};
    color: {TEXT_DIM};
    font-weight: 500;
    font-size: 11px;
    letter-spacing: .05em;
    text-transform: uppercase;
    padding: 10px 14px;
    border-bottom: 1px solid {BORDER};
    text-align: left;
  }}
  .styled-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid {BORDER}88;
    color: {TEXT};
  }}
  .styled-table tr:last-child td {{ border-bottom: none; }}
  .styled-table tr:hover td {{ background: {BG_CARD2}; }}

  /* Info banner */
  .info-banner {{
    background: {ACCENT}11;
    border: 1px solid {ACCENT}33;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: {TEXT_DIM};
    margin: 12px 0;
  }}
  .warn-banner {{
    background: {YELLOW}11;
    border: 1px solid {YELLOW}44;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    color: {YELLOW};
    margin-bottom: 16px;
  }}

  /* Divider */
  hr {{ border-color: {BORDER}; margin: 20px 0; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {BORDER};
    gap: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {TEXT_DIM};
    font-size: 13px;
    font-weight: 500;
    padding: 10px 20px;
    border: none;
    border-bottom: 2px solid transparent;
  }}
  .stTabs [aria-selected="true"] {{
    color: {TEXT} !important;
    border-bottom: 2px solid {ACCENT} !important;
    background: transparent !important;
  }}

  /* Hide Streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
AAPL_FALLBACK = {
    "base_fcf":   108_807_000_000,
    "net_debt":    67_874_000_000,
    "beta":                  1.24,
    "shares":     15_115_823_000,
    "current_price":         None,
}

@st.cache_data(ttl=3600)
def load_apple_data():
    try:
        ticker   = yf.Ticker("AAPL")
        cashflow = ticker.cashflow
        balance  = ticker.balance_sheet
        info     = ticker.info
        base_fcf = (float(cashflow.loc["Free Cash Flow"].iloc[0])
                    if "Free Cash Flow" in cashflow.index
                    else AAPL_FALLBACK["base_fcf"])
        try:
            td = float(balance.loc["Total Debt"].iloc[0]) if "Total Debt" in balance.index else 0
            c  = float(balance.loc["Cash And Cash Equivalents"].iloc[0]) if "Cash And Cash Equivalents" in balance.index else 0
            net_debt = td - c
        except Exception:
            net_debt = AAPL_FALLBACK["net_debt"]
        return {"base_fcf": base_fcf, "net_debt": net_debt,
                "beta":   info.get("beta")              or AAPL_FALLBACK["beta"],
                "shares": info.get("sharesOutstanding") or AAPL_FALLBACK["shares"],
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "source": "live"}
    except Exception:
        return {**AAPL_FALLBACK, "source": "fallback"}


@st.cache_data(ttl=3600)
def load_risk_free_rate():
    try:
        fred  = Fred(api_key=st.secrets["FRED_API_KEY"])
        dgs10 = fred.get_series("DGS10", observation_start="2024-01-01").dropna()
        return float(dgs10.iloc[-1]) / 100
    except Exception:
        return 0.045


# ── Plotly chart builders ─────────────────────────────────────────────────────
def fcf_bar(base_fcf, growth_rates):
    fcfs   = [base_fcf]
    for g in growth_rates:
        fcfs.append(fcfs[-1] * (1 + g))
    labels = ["FY0 (Actual)"] + [f"Y+{i+1}" for i in range(len(growth_rates))]
    colors = [TEXT_DIM] + [ACCENT] * len(growth_rates)

    fig = go.Figure(go.Bar(
        x=labels, y=[v / 1e9 for v in fcfs],
        marker_color=colors,
        text=[f"${v/1e9:.0f}B" for v in fcfs],
        textposition="outside",
        textfont=dict(size=11, color=TEXT),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Free Cash Flow Projection", font=dict(size=14, color=TEXT), x=0),
        xaxis=XAXIS, yaxis={**YAXIS, "title": "Billion USD"},
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
    )
    return fig


def ev_donut(sum_pv_fcfs, pv_tv):
    fig = go.Figure(go.Pie(
        labels=["PV(FCF 1–5)", "PV(Terminal Value)"],
        values=[sum_pv_fcfs, pv_tv],
        hole=0.6,
        marker=dict(colors=[ACCENT, GREEN],
                    line=dict(color=BG_CARD, width=3)),
        textfont=dict(size=12, color=TEXT),
        hovertemplate="%{label}<br>$%{value:.0f}<extra></extra>",
    ))
    fig.add_annotation(text="EV Split", x=0.5, y=0.5,
                       font=dict(size=13, color=TEXT_DIM), showarrow=False)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Enterprise Value Breakdown", font=dict(size=14, color=TEXT), x=0),
        height=340, showlegend=True,
        legend=dict(font=dict(color=TEXT_DIM), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def scenario_bar(dcf_results, current_price):
    names  = ["Bear", "Base", "Bull"]
    values = [dcf_results[n]["intrinsic_value_per_share"] for n in names]
    colors = [RED, ACCENT, GREEN]

    fig = go.Figure()
    for name, val, col in zip(names, values, colors):
        fig.add_trace(go.Bar(
            name=name, x=[name], y=[val],
            marker_color=col,
            marker_line_width=0,
            text=f"${val:.0f}",
            textposition="outside",
            textfont=dict(size=14, color=TEXT, family=FONT),
            width=0.45,
        ))
    if current_price:
        fig.add_hline(y=current_price, line_dash="dot",
                      line_color=YELLOW, line_width=1.5,
                      annotation_text=f"  Market ${current_price:.0f}",
                      annotation_font_color=YELLOW,
                      annotation_font_size=11)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Intrinsic Value by Scenario", font=dict(size=14, color=TEXT), x=0),
        xaxis=XAXIS, yaxis={**YAXIS, "title": "Intrinsic Value per Share ($)"},
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
        bargap=0.5,
    )
    return fig


def valuation_range(dcf_results, current_price):
    bull = dcf_results["Bull"]["intrinsic_value_per_share"]
    base = dcf_results["Base"]["intrinsic_value_per_share"]
    bear = dcf_results["Bear"]["intrinsic_value_per_share"]

    fig = go.Figure()
    # Range bar
    fig.add_trace(go.Bar(
        x=[bull - bear], y=["Valuation Range"],
        base=bear, orientation="h",
        marker=dict(color=ACCENT, opacity=0.15, line=dict(color=ACCENT, width=1)),
        showlegend=False, hoverinfo="skip",
    ))
    # Points
    for val, col, name in [(bear, RED, "Bear"), (base, ACCENT, "Base"), (bull, GREEN, "Bull")]:
        fig.add_trace(go.Scatter(
            x=[val], y=["Valuation Range"],
            mode="markers+text",
            marker=dict(symbol="diamond", size=14, color=col,
                        line=dict(color=BG_CARD, width=2)),
            text=[f"<b>${val:.0f}</b>"],
            textposition="top center",
            textfont=dict(color=col, size=11),
            name=name,
        ))
    if current_price:
        fig.add_vline(x=current_price, line_dash="dot", line_color=YELLOW,
                      line_width=1.5,
                      annotation_text=f" Market ${current_price:.0f}",
                      annotation_font_color=YELLOW, annotation_font_size=11,
                      annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Bull / Base / Bear Range", font=dict(size=14, color=TEXT), x=0),
        height=200,
        xaxis={**XAXIS, "title": "Intrinsic Value per Share ($)"},
        yaxis=dict(showticklabels=False, showgrid=False, color=TEXT_DIM),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25,
                    font=dict(color=TEXT_DIM), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=80, t=40, b=40),
    )
    return fig


def sensitivity_heatmap(base_fcf, growth_rates, net_debt, shares, beta):
    tbl = sensitivity_table(base_fcf, growth_rates, net_debt, shares, beta)
    z   = tbl.values.astype(float)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=tbl.columns.tolist(),
        y=tbl.index.tolist(),
        colorscale=[[0,"#7f1d1d"],[0.35,"#b45309"],[0.6,"#ca8a04"],
                    [0.8,"#16a34a"],[1,"#166534"]],
        text=[[f"${v:.0f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        showscale=True,
        colorbar=dict(tickfont=dict(color=TEXT_DIM), bgcolor="rgba(0,0,0,0)"),
        hovertemplate="WACC: %{y}<br>TG: %{x}<br>Value: %{text}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Intrinsic Value: WACC × Terminal Growth", font=dict(size=14, color=TEXT), x=0),
        xaxis={**XAXIS, "title": "Terminal Growth Rate"},
        yaxis=dict(autorange="reversed", showgrid=False, color=TEXT_DIM,
                   tickfont=dict(color=TEXT_DIM), title="WACC"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=380,
    )
    return fig


def tornado_chart(t_df, base_val):
    fig = go.Figure()
    for _, row in t_df.iterrows():
        fig.add_trace(go.Bar(
            name="Low",
            x=[row["Low ($)"] - base_val],
            y=[row["Variable"]],
            base=base_val,
            orientation="h",
            marker_color=RED,
            marker_line_width=0,
            text=f"${row['Low ($)']:.0f}",
            textposition="inside",
            textfont=dict(color="white", size=10),
            showlegend=False,
            hovertemplate=f"Low: ${row['Low ($)']:.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="High",
            x=[row["High ($)"] - base_val],
            y=[row["Variable"]],
            base=base_val,
            orientation="h",
            marker_color=GREEN,
            marker_line_width=0,
            text=f"${row['High ($)']:.0f}",
            textposition="inside",
            textfont=dict(color="white", size=10),
            showlegend=False,
            hovertemplate=f"High: ${row['High ($)']:.0f}<extra></extra>",
        ))
    fig.add_vline(x=base_val, line_dash="dot", line_color=ACCENT, line_width=1.5,
                  annotation_text=f" Base ${base_val:.0f}",
                  annotation_font_color=ACCENT, annotation_font_size=11)
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Tornado: Value Drivers Impact", font=dict(size=14, color=TEXT), x=0),
        xaxis={**XAXIS, "title": "Intrinsic Value per Share ($)"},
        yaxis={**YAXIS, "showgrid": False},
        barmode="overlay",
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
    )
    return fig


# ── KPI card helper ───────────────────────────────────────────────────────────
def kpi(label, value, sub="", cls="kpi-neu"):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub {cls}">{sub}</div>
    </div>"""


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 16px">
      <div style="font-size:16px;font-weight:700;color:{TEXT}">DCF Parameters</div>
      <div style="font-size:11px;color:{TEXT_DIM};margin-top:2px">Apple · AAPL · FY2024</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:{TEXT_DIM};margin-bottom:8px'>FCF Growth Rates</div>", unsafe_allow_html=True)
    y1 = st.slider("Year 1", 1, 20, 8, format="%d%%") / 100
    y2 = st.slider("Year 2", 1, 20, 8, format="%d%%") / 100
    y3 = st.slider("Year 3", 1, 20, 7, format="%d%%") / 100
    y4 = st.slider("Year 4", 1, 20, 7, format="%d%%") / 100
    y5 = st.slider("Year 5", 1, 20, 6, format="%d%%") / 100
    growth_rates = [y1, y2, y3, y4, y5]

    st.markdown(f"<div style='font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:{TEXT_DIM};margin:16px 0 8px'>Model Assumptions</div>", unsafe_allow_html=True)
    tg  = st.slider("Terminal Growth", 1, 5,  3, format="%d%%") / 100
    dw  = st.slider("Debt Weight",     0, 50, 25, format="%d%%") / 100
    cod = st.slider("Cost of Debt",    1, 8,  35, format="%d%%") / 100 / 10

    st.markdown("<hr>", unsafe_allow_html=True)
    st.button("↻  Recalculate", use_container_width=True, type="primary")


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
data = load_apple_data()
rf   = load_risk_free_rate()

base_fcf = data["base_fcf"]
net_debt = data["net_debt"]
beta     = data["beta"]
shares   = data["shares"]
price    = data["current_price"]

wacc        = calculate_wacc(rf, beta, MARKET_RISK_PREMIUM, dw, cod)
base_result = intrinsic_value_per_share(base_fcf, growth_rates, wacc, tg, net_debt, shares)
iv          = base_result["intrinsic_value_per_share"]

dcf_results = {}
for name, sc in SCENARIOS.items():
    w = calculate_wacc(sc["risk_free_rate"], beta, MARKET_RISK_PREMIUM)
    dcf_results[name] = intrinsic_value_per_share(
        base_fcf, sc["fcf_growth_rates"], w, sc["terminal_growth"], net_debt, shares)

bull_v = dcf_results["Bull"]["intrinsic_value_per_share"]
base_v = dcf_results["Base"]["intrinsic_value_per_share"]
bear_v = dcf_results["Bear"]["intrinsic_value_per_share"]


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:4px">
  <span class="page-title">Apple Inc. — DCF Valuation</span>
  <span class="ticker-badge">AAPL · NASDAQ</span>
</div>
<div class="page-subtitle">
  Macro-driven scenario analysis · FRED API + Yahoo Finance · FP&A Portfolio
</div>
""", unsafe_allow_html=True)

if data.get("source") == "fallback":
    st.markdown(f'<div class="warn-banner">⚠ Live data unavailable (Yahoo Finance rate limit). Using Apple FY2024 actuals — FCF $108.8B.</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# KPI row
updown = ((iv / price) - 1) * 100 if price else None
cls    = "kpi-up" if updown and updown > 0 else "kpi-down"
sub    = f"{updown:+.1f}% vs market" if updown else "live price unavailable"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(kpi("Base Intrinsic Value", f"${iv:.2f}", sub, cls), unsafe_allow_html=True)
with c2:
    st.markdown(kpi("Market Price", f"${price:.2f}" if price else "—",
                    "Yahoo Finance · real-time"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi("WACC", f"{wacc:.2%}", f"Rf {rf:.2%} · β {beta:.2f}"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi("Bull / Bear Spread", f"${bull_v - bear_v:.0f}",
                    f"${bear_v:.0f}  ←→  ${bull_v:.0f}", "kpi-neu"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  📊  DCF Model  ", "  🌍  Scenario Analysis  ", "  🔥  Sensitivity  "])

# ── TAB 1: DCF ────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fcf_bar(base_fcf, growth_rates), use_container_width=True)
    with c2:
        st.plotly_chart(ev_donut(base_result["sum_pv_fcfs"], base_result["pv_terminal_value"]),
                        use_container_width=True)

    st.markdown('<div class="section-header">Valuation Summary</div>', unsafe_allow_html=True)

    rows = [
        ("Risk-Free Rate (10Y Treasury)", f"{rf:.2%}"),
        ("Beta (5Y Monthly)",             f"{beta:.2f}"),
        ("WACC",                          f"{wacc:.2%}"),
        ("Terminal Growth Rate",          f"{tg:.1%}"),
        ("FCF PV Sum (5Y)",               f"${base_result['sum_pv_fcfs']/1e9:.1f}B"),
        ("PV(Terminal Value)",            f"${base_result['pv_terminal_value']/1e9:.1f}B"),
        ("Enterprise Value",              f"${base_result['enterprise_value']/1e9:.1f}B"),
        ("(−) Net Debt",                  f"${net_debt/1e9:.1f}B"),
        ("Equity Value",                  f"${base_result['equity_value']/1e9:.1f}B"),
        ("Intrinsic Value / Share",       f"<b style='color:{ACCENT}'>${iv:.2f}</b>"),
    ]
    html = '<table class="styled-table">'
    html += "<thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>"
    for k, v in rows:
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


# ── TAB 2: SCENARIOS ──────────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(scenario_bar(dcf_results, price), use_container_width=True)
    with c2:
        st.plotly_chart(valuation_range(dcf_results, price), use_container_width=True)

    st.markdown('<div class="section-header">Scenario Parameters</div>', unsafe_allow_html=True)

    summary = build_scenario_summary(dcf_results).reset_index()
    html = '<table class="styled-table"><thead><tr>'
    for col in summary.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    colors_map = {"Bull Case": GREEN, "Base Case": ACCENT, "Bear Case": RED}
    for _, row in summary.iterrows():
        col = colors_map.get(row["Scenario"], TEXT)
        html += f"<tr><td><b style='color:{col}'>{row['Scenario']}</b></td>"
        for val in row[1:]:
            html += f"<td>{val}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

    spread_pct = (bull_v / bear_v - 1) * 100
    st.markdown(f"""
    <div class="info-banner">
      📌  Bull/Bear spread: <b style="color:{TEXT}">${bull_v - bear_v:.0f} per share</b>
      ({spread_pct:.0f}% premium) — quantifying macroeconomic uncertainty on corporate valuation.
    </div>""", unsafe_allow_html=True)


# ── TAB 3: SENSITIVITY ────────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            sensitivity_heatmap(base_fcf, growth_rates, net_debt, shares, beta),
            use_container_width=True)
    with c2:
        try:
            t_df = tornado_inputs(base_fcf, base_result, net_debt, shares, beta)
            st.plotly_chart(tornado_chart(t_df, iv), use_container_width=True)
        except Exception as e:
            st.warning(f"Tornado error: {e}")

    st.markdown(f"""
    <div style="font-size:11px; color:{TEXT_DIM}; margin-top:8px">
      Each heatmap cell = intrinsic value per share ($) at the given WACC × terminal growth combination.
      Tornado shows isolated impact of each variable (±1 standard deviation from base).
    </div>""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<hr>
<div style="display:flex; justify-content:space-between; font-size:11px; color:{TEXT_DIM}; padding:4px 0">
  <span>Data: Yahoo Finance · FRED API (Federal Reserve) · Methodology: CAPM WACC · Gordon Growth Model</span>
  <a href="https://github.com/Tuluntas09/python-apple-dcf-analysis"
     style="color:{ACCENT}; text-decoration:none">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
