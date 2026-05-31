"""
Apple DCF Analysis — Interactive Streamlit Dashboard
Portfolio project: Financial Analyst / FP&A
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from fredapi import Fred

from src.dcf_model import (
    calculate_wacc, intrinsic_value_per_share,
    sensitivity_table, MARKET_RISK_PREMIUM
)
from src.scenario_engine import SCENARIOS, build_scenario_summary, tornado_inputs

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple DCF Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Apple finansal verileri yükleniyor...")
def load_apple_data():
    ticker = yf.Ticker("AAPL")
    cashflow = ticker.cashflow
    balance  = ticker.balance_sheet
    info     = ticker.info

    # FCF
    if "Free Cash Flow" in cashflow.index:
        base_fcf = float(cashflow.loc["Free Cash Flow"].iloc[0])
    else:
        base_fcf = 99_584_000_000

    # Net debt
    try:
        total_debt = float(balance.loc["Total Debt"].iloc[0]) if "Total Debt" in balance.index else 0
        cash       = float(balance.loc["Cash And Cash Equivalents"].iloc[0]) if "Cash And Cash Equivalents" in balance.index else 0
        net_debt   = total_debt - cash
    except Exception:
        net_debt = 36_000_000_000

    beta   = info.get("beta", 1.24)
    shares = info.get("sharesOutstanding", 15_500_000_000)
    price  = info.get("currentPrice") or info.get("regularMarketPrice")

    return {"base_fcf": base_fcf, "net_debt": net_debt,
            "beta": beta, "shares": shares, "current_price": price}


@st.cache_data(ttl=3600, show_spinner="FRED makro verileri yükleniyor...")
def load_risk_free_rate():
    try:
        fred = Fred(api_key=st.secrets["FRED_API_KEY"])
        dgs10 = fred.get_series("DGS10", observation_start="2024-01-01").dropna()
        return float(dgs10.iloc[-1]) / 100
    except Exception:
        return 0.045


def make_fcf_chart(base_fcf, growth_rates):
    fcfs = [base_fcf]
    for g in growth_rates:
        fcfs.append(fcfs[-1] * (1 + g))
    labels = ["FY0"] + [f"Y+{i+1}" for i in range(len(growth_rates))]
    colors = ["#95a5a6"] + ["#3498db"] * len(growth_rates)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, [v / 1e9 for v in fcfs], color=colors, alpha=0.85)
    for bar, val in zip(bars, fcfs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"${val/1e9:.0f}B", ha="center", fontsize=9)
    ax.set_ylabel("Milyar USD")
    ax.set_title("FCF Projeksiyonu (5 Yıl)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def make_ev_pie(sum_pv_fcfs, pv_tv):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        [sum_pv_fcfs, pv_tv],
        labels=["PV(FCF 1-5)", "PV(Terminal Value)"],
        colors=["#3498db", "#2ecc71"],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title("Enterprise Value Bileşenleri", fontweight="bold")
    plt.tight_layout()
    return fig


def make_scenario_chart(dcf_results, current_price):
    names  = list(SCENARIOS.keys())
    values = [dcf_results[n]["intrinsic_value_per_share"] for n in names]
    colors = [SCENARIOS[n]["color"] for n in names]
    labels = [SCENARIOS[n]["label"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"${val:.0f}", ha="center", fontsize=12, fontweight="bold")
    if current_price:
        ax.axhline(current_price, color="black", linewidth=1.5,
                   linestyle="--", label=f"Piyasa Fiyatı (${current_price:.0f})")
        ax.legend(fontsize=10)
    ax.set_ylabel("Hisse Başına İçsel Değer ($)")
    ax.set_title("Senaryo Bazlı DCF Değerlemesi", fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def make_sensitivity_heatmap(base_fcf, growth_rates, net_debt, shares, beta):
    tbl = sensitivity_table(base_fcf, growth_rates, net_debt, shares, beta)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(tbl.astype(float), annot=True, fmt=".0f",
                cmap="RdYlGn", ax=ax, linewidths=0.5,
                annot_kws={"size": 10})
    ax.set_title("Sensitivite: Hisse Başına İçsel Değer ($)\nWACC × Terminal Büyüme Oranı",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Terminal Büyüme Oranı")
    ax.set_ylabel("WACC")
    plt.tight_layout()
    return fig


def make_tornado_chart(tornado_df, base_val):
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, row in tornado_df.reset_index(drop=True).iterrows():
        ax.barh(i, row["Low ($)"]  - base_val, left=base_val, color="#e74c3c", alpha=0.8, height=0.5)
        ax.barh(i, row["High ($)"] - base_val, left=base_val, color="#2ecc71", alpha=0.8, height=0.5)
        ax.text(row["Low ($)"]  - 0.5, i, f"${row['Low ($)']:.0f}",  ha="right", va="center", fontsize=9)
        ax.text(row["High ($)"] + 0.5, i, f"${row['High ($)']:.0f}", ha="left",  va="center", fontsize=9)
    ax.set_yticks(range(len(tornado_df)))
    ax.set_yticklabels(tornado_df["Variable"].values)
    ax.axvline(base_val, color="black", linewidth=1.5, linestyle="--",
               label=f"Base: ${base_val:.0f}")
    red_p   = mpatches.Patch(color="#e74c3c", alpha=0.8, label="Düşük Senaryo")
    green_p = mpatches.Patch(color="#2ecc71", alpha=0.8, label="Yüksek Senaryo")
    ax.legend(handles=[red_p, green_p], fontsize=9)
    ax.set_xlabel("Hisse Başına İçsel Değer ($)")
    ax.set_title("Tornado Chart — En Kritik Değer Sürücüleri", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parametreler")
    st.caption("Apple (AAPL) — makro veriler FRED API'den, finansallar Yahoo Finance'ten çekilir.")

    st.subheader("📈 FCF Büyüme Oranları")
    y1 = st.slider("Y+1 (%)", 1, 20, 8) / 100
    y2 = st.slider("Y+2 (%)", 1, 20, 8) / 100
    y3 = st.slider("Y+3 (%)", 1, 20, 7) / 100
    y4 = st.slider("Y+4 (%)", 1, 20, 7) / 100
    y5 = st.slider("Y+5 (%)", 1, 20, 6) / 100
    growth_rates = [y1, y2, y3, y4, y5]

    st.subheader("🏦 DCF Varsayımları")
    tg  = st.slider("Terminal Growth (%)", 1, 5, 3) / 100
    dw  = st.slider("Borç Ağırlığı (%)", 0, 50, 25) / 100
    cod = st.slider("Borç Maliyeti (%)", 1, 8, 35) / 100 / 10  # 3.5% default

    st.markdown("---")
    run = st.button("🔄 Hesapla", use_container_width=True, type="primary")

# ── Load data ─────────────────────────────────────────────────────────────────
data = load_apple_data()
rf   = load_risk_free_rate()

base_fcf = data["base_fcf"]
net_debt = data["net_debt"]
beta     = data["beta"]
shares   = data["shares"]
price    = data["current_price"]

wacc = calculate_wacc(rf, beta, MARKET_RISK_PREMIUM, dw, cod)
base_result = intrinsic_value_per_share(
    base_fcf, growth_rates, wacc, tg, net_debt, shares
)
iv = base_result["intrinsic_value_per_share"]

# Senaryo sonuçları
dcf_results = {}
for name, sc in SCENARIOS.items():
    w = calculate_wacc(sc["risk_free_rate"], beta, MARKET_RISK_PREMIUM)
    dcf_results[name] = intrinsic_value_per_share(
        base_fcf, sc["fcf_growth_rates"], w,
        sc["terminal_growth"], net_debt, shares
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Apple (AAPL) — DCF Değerleme Analizi")
st.caption("Makroekonomik senaryo analizi | FRED API + Yahoo Finance | FP&A Portfolio Projesi")

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Base İçsel Değer",  f"${iv:.2f}")
col2.metric("📈 Piyasa Fiyatı",     f"${price:.2f}" if price else "—")
col3.metric("⚖️ Fark",
            f"{((iv/price)-1)*100:+.1f}%" if price else "—",
            delta_color="normal")
col4.metric("📉 WACC",             f"{wacc:.2%}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 DCF Model", "🌍 Senaryo Analizi", "🔥 Sensitivite"])

# ── Tab 1: DCF ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("DCF Model — Base Case")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.pyplot(make_fcf_chart(base_fcf, growth_rates))
    with c2:
        st.pyplot(make_ev_pie(base_result["sum_pv_fcfs"], base_result["pv_terminal_value"]))

    st.markdown("#### Değerleme Özeti")
    summary = pd.DataFrame({
        "Parametre": ["Risk-Free Rate", "Beta", "WACC", "Terminal Growth",
                      "FCF PV Toplamı", "Terminal Value (PV)",
                      "Enterprise Value", "Equity Value", "İçsel Değer / Hisse"],
        "Değer": [
            f"{rf:.2%}", f"{beta:.2f}", f"{wacc:.2%}", f"{tg:.1%}",
            f"${base_result['sum_pv_fcfs']/1e9:.1f}B",
            f"${base_result['pv_terminal_value']/1e9:.1f}B",
            f"${base_result['enterprise_value']/1e9:.1f}B",
            f"${base_result['equity_value']/1e9:.1f}B",
            f"**${iv:.2f}**",
        ]
    }).set_index("Parametre")
    st.dataframe(summary, use_container_width=True)

# ── Tab 2: Scenarios ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Bull / Base / Bear Senaryo Analizi")

    bull_v = dcf_results["Bull"]["intrinsic_value_per_share"]
    bear_v = dcf_results["Bear"]["intrinsic_value_per_share"]
    base_v = dcf_results["Base"]["intrinsic_value_per_share"]

    c1, c2, c3 = st.columns(3)
    c1.metric("🐂 Bull Case", f"${bull_v:.0f}",
              delta=f"+{((bull_v/price)-1)*100:.0f}% piyasaya göre" if price else None)
    c2.metric("📊 Base Case", f"${base_v:.0f}",
              delta=f"{((base_v/price)-1)*100:.0f}% piyasaya göre" if price else None)
    c3.metric("🐻 Bear Case", f"${bear_v:.0f}",
              delta=f"{((bear_v/price)-1)*100:.0f}% piyasaya göre" if price else None,
              delta_color="inverse")

    st.pyplot(make_scenario_chart(dcf_results, price))

    st.markdown("#### Senaryo Parametreleri")
    st.dataframe(build_scenario_summary(dcf_results), use_container_width=True)

    st.info(f"📌 Bull/Bear fark: **${bull_v - bear_v:.0f}** ({(bull_v/bear_v-1)*100:.0f}% spread) — "
            f"makro belirsizliğin değerlemeye etkisi")

# ── Tab 3: Sensitivity ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Sensitivite & Tornado Analizi")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### WACC × Terminal Growth Heatmap")
        st.pyplot(make_sensitivity_heatmap(base_fcf, growth_rates, net_debt, shares, beta))
    with c2:
        st.markdown("##### Tornado Chart — En Kritik Değişkenler")
        try:
            t_df = tornado_inputs(base_fcf, base_result, net_debt, shares, beta)
            st.pyplot(make_tornado_chart(t_df, iv))
        except Exception as e:
            st.warning(f"Tornado hesabı: {e}")

    st.markdown("---")
    st.caption("Her hücre: belirtilen WACC ve terminal growth kombinasyonunda hisse başına içsel değer ($)")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Veri kaynakları: Yahoo Finance (yfinance) · FRED API (Federal Reserve) · "
    "Metodoloji: CAPM tabanlı WACC · Gordon Growth Model terminal value · "
    "[GitHub](https://github.com/Tuluntas09/python-apple-dcf-analysis)"
)
