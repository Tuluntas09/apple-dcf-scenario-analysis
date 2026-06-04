from __future__ import annotations

from datetime import date, timedelta
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.advanced_analytics import (
    rebalancing_comparison,
    rolling_metrics,
    stress_test_summary,
    transaction_cost_impact,
)
from src.data_loader import DEFAULT_TICKERS, load_prices, parse_tickers
from src.metrics import (
    asset_metric_table,
    benchmark_comparison,
    calculate_returns,
    concentration_summary,
    cumulative_growth,
    portfolio_metric_summary,
    portfolio_returns,
    risk_contribution,
)
from src.optimization import optimize_portfolio, portfolio_performance, random_portfolios
from src.portfolio_inputs import lots_to_position_values, lots_to_weights
from src.providers.finnhub import fetch_finnhub_snapshot, has_finnhub_key
from src.reporting import build_html_report, dataframe_to_csv_bytes
from src.simulation import monte_carlo_paths, terminal_value_summary
from src.validation import concentration_warnings, validate_clean_dataset, validate_requested_inputs


st.set_page_config(
    page_title="Portfoy Analizi",
    page_icon="",
    layout="wide",
)


PRIMARY_COLOR = "#2563eb"
ACCENT_COLOR = "#16a34a"
WARNING_COLOR = "#dc2626"

PRESET_PORTFOLIOS = {
    "Esit agirlik": None,
    "Dengeli": {"SPY": 0.35, "QQQ": 0.25, "TLT": 0.25, "GLD": 0.15},
    "Buyume": {"SPY": 0.35, "QQQ": 0.45, "TLT": 0.10, "GLD": 0.10},
    "Defansif": {"SPY": 0.25, "QQQ": 0.15, "TLT": 0.45, "GLD": 0.15},
    "Enflasyon koruma": {"SPY": 0.25, "QQQ": 0.20, "TLT": 0.15, "GLD": 0.40},
}

ANALYSIS_PROFILES = {
    "Dengeli analiz": {
        "description": "Getiri, risk ve optimizasyon metriklerini dengeli yorumlar.",
        "risk_free_rate": 0.04,
        "confidence": 0.95,
        "initial_value": 10_000,
        "transaction_cost_bps": 10,
    },
    "Risk odakli analiz": {
        "description": "Kayıp olasılığı, volatilite ve aşağı yönlü riskleri daha sıkı değerlendirir.",
        "risk_free_rate": 0.04,
        "confidence": 0.99,
        "initial_value": 10_000,
        "transaction_cost_bps": 15,
    },
    "Getiri odakli analiz": {
        "description": "Risk sınırlarını korurken getiri ve optimizasyon fırsatlarını öne çıkarır.",
        "risk_free_rate": 0.03,
        "confidence": 0.95,
        "initial_value": 10_000,
        "transaction_cost_bps": 5,
    },
}

with st.sidebar:
    theme_mode = st.radio("Tema", ["Acik", "Koyu"], horizontal=True, key="theme_choice")

THEMES = {
    "Acik": {
        "app_bg": "#f8fafc",
        "panel_bg": "#ffffff",
        "card_bg": "#ffffff",
        "input_bg": "#ffffff",
        "text": "#111827",
        "muted": "#64748b",
        "border": "#e5e7eb",
        "grid": "#e5e7eb",
        "soft": "#f1f5f9",
        "plot_template": "plotly_white",
    },
    "Koyu": {
        "app_bg": "#020617",
        "panel_bg": "#050b18",
        "card_bg": "#0b1220",
        "input_bg": "#111827",
        "text": "#f8fafc",
        "muted": "#cbd5e1",
        "border": "#243244",
        "grid": "#1f2937",
        "soft": "#0f172a",
        "plot_template": "plotly_dark",
    },
}
ACTIVE_THEME = THEMES[theme_mode]


st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: {ACTIVE_THEME["app_bg"]};
        color: {ACTIVE_THEME["text"]};
    }}

    [data-testid="stHeader"],
    [data-testid="stToolbar"] {{
        background: {ACTIVE_THEME["app_bg"]} !important;
        color: {ACTIVE_THEME["text"]} !important;
        border: 0 !important;
        border-color: {ACTIVE_THEME["app_bg"]} !important;
        box-shadow: none !important;
    }}

    [data-testid="stHeader"]::before {{
        background: {ACTIVE_THEME["app_bg"]} !important;
    }}

    body,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    label,
    p {{
        color: {ACTIVE_THEME["text"]} !important;
    }}

    [data-testid="stSidebar"] {{
        background: {ACTIVE_THEME["panel_bg"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {ACTIVE_THEME["text"]};
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    [data-testid="stMetric"] {{
        background: {ACTIVE_THEME["card_bg"]};
        border: 1px solid {ACTIVE_THEME["border"]};
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}

    [data-testid="stMetricLabel"] {{
        color: {ACTIVE_THEME["muted"]};
        font-size: 0.88rem;
    }}

    [data-testid="stMetricValue"] {{
        color: {ACTIVE_THEME["text"]};
        font-size: 1.75rem;
        line-height: 1.2;
    }}

    .small-muted {{
        color: {ACTIVE_THEME["muted"]};
        font-size: 0.9rem;
    }}

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] {{
        background-color: {ACTIVE_THEME["input_bg"]} !important;
        border-color: {ACTIVE_THEME["border"]} !important;
        color: {ACTIVE_THEME["text"]} !important;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        color: {ACTIVE_THEME["muted"]} !important;
    }}

    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {{
        background-color: {ACTIVE_THEME["soft"]} !important;
        border: 1px solid {ACTIVE_THEME["border"]} !important;
        color: {ACTIVE_THEME["text"]} !important;
        border-radius: 8px !important;
    }}

    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-primary"] * {{
        color: {ACTIVE_THEME["text"]} !important;
    }}

    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
        background: transparent !important;
        border: 0 !important;
        min-height: 28px !important;
        padding: 0 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * {{
        color: {ACTIVE_THEME["muted"]} !important;
        font-size: 1rem !important;
        line-height: 1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover * {{
        color: {WARNING_COLOR} !important;
    }}

    .portfolio-asset-label {{
        color: {ACTIVE_THEME["text"]};
        font-size: 0.95rem;
        line-height: 1.8rem;
        padding: 0;
    }}

    [data-testid="stExpander"] {{
        background-color: {ACTIVE_THEME["card_bg"]} !important;
        border: 1px solid {ACTIVE_THEME["border"]} !important;
        border-radius: 8px !important;
    }}

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary:hover {{
        background-color: {ACTIVE_THEME["card_bg"]} !important;
        color: {ACTIVE_THEME["text"]} !important;
        border-color: {ACTIVE_THEME["border"]} !important;
    }}

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary svg {{
        color: {ACTIVE_THEME["text"]} !important;
        fill: {ACTIVE_THEME["text"]} !important;
    }}

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        background-color: {ACTIVE_THEME["card_bg"]} !important;
        color: {ACTIVE_THEME["text"]} !important;
        border: 1px solid {ACTIVE_THEME["border"]} !important;
        border-radius: 8px !important;
        overflow: hidden;
    }}

    [data-testid="stDataFrame"] [data-testid="stElementToolbarButtonContainer"],
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
    [data-testid="stDataFrame"] [data-testid="stBaseButton-elementToolbar"] {{
        background-color: {ACTIVE_THEME["card_bg"]} !important;
        color: {ACTIVE_THEME["text"]} !important;
    }}

    [data-testid="stDataFrame"] canvas {{
        background-color: {ACTIVE_THEME["card_bg"]} !important;
    }}

    .data-table-wrap {{
        max-height: 420px;
        overflow: auto;
        border: 1px solid {ACTIVE_THEME["border"]};
        border-radius: 8px;
        background: {ACTIVE_THEME["card_bg"]};
        margin: 0.65rem 0 1.05rem 0;
    }}

    .data-table {{
        width: 100%;
        border-collapse: collapse;
        color: {ACTIVE_THEME["text"]};
        font-size: 0.88rem;
        border-color: {ACTIVE_THEME["border"]} !important;
    }}

    .data-table thead,
    .data-table tbody,
    .data-table tr {{
        background: {ACTIVE_THEME["card_bg"]};
        border-color: {ACTIVE_THEME["border"]} !important;
    }}

    .data-table th {{
        position: sticky;
        top: 0;
        z-index: 1;
        background: {ACTIVE_THEME["soft"]};
        color: {ACTIVE_THEME["muted"]};
        font-weight: 600;
        text-align: left;
        border-bottom: 1px solid {ACTIVE_THEME["border"]};
        border-right: 1px solid {ACTIVE_THEME["border"]};
        border-color: {ACTIVE_THEME["border"]} !important;
        padding: 10px 12px;
        white-space: nowrap;
    }}

    .data-table td {{
        background: {ACTIVE_THEME["card_bg"]};
        color: {ACTIVE_THEME["text"]};
        border-bottom: 1px solid {ACTIVE_THEME["border"]};
        border-right: 1px solid {ACTIVE_THEME["border"]};
        border-color: {ACTIVE_THEME["border"]} !important;
        padding: 10px 12px;
        vertical-align: top;
    }}

    .data-table tr:last-child td {{
        border-bottom: 0;
    }}

    .data-table th:last-child,
    .data-table td:last-child {{
        border-right: 0;
    }}

    .data-table .numeric-cell {{
        text-align: right;
        font-variant-numeric: tabular-nums;
    }}

    [data-testid="stSidebar"] .data-table {{
        font-size: 0.78rem;
    }}

    [data-testid="stSidebar"] .data-table th,
    [data-testid="stSidebar"] .data-table td {{
        padding: 8px 6px;
        white-space: nowrap;
    }}

    [data-testid="stAlert"] {{
        background-color: {ACTIVE_THEME["soft"]} !important;
        color: {ACTIVE_THEME["text"]} !important;
        border-color: {ACTIVE_THEME["border"]} !important;
    }}

    .theme-card {{
        background: {ACTIVE_THEME["card_bg"]};
        border: 1px solid {ACTIVE_THEME["border"]};
        border-radius: 8px;
        padding: 16px;
    }}

    .theme-muted {{
        color: {ACTIVE_THEME["muted"]};
        font-size: 0.92rem;
    }}

    .sidebar-kicker {{
        color: {ACTIVE_THEME["muted"]};
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.15rem;
    }}

    .sidebar-status {{
        background: {ACTIVE_THEME["soft"]};
        border: 1px solid {ACTIVE_THEME["border"]};
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0 12px 0;
        color: {ACTIVE_THEME["text"]};
        font-size: 0.9rem;
    }}

    @media (max-width: 900px) {{
        .decision-grid {{
            grid-template-columns: 1fr !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.2f}"


def format_money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1%}"


def format_signed_number(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}"


def clean_chart(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template=ACTIVE_THEME["plot_template"],
        height=height,
        margin={"l": 20, "r": 20, "t": 54, "b": 28},
        legend_title_text="",
        hovermode="x unified",
        font={"family": "Arial, sans-serif", "size": 13, "color": ACTIVE_THEME["text"]},
        title={"font": {"size": 18}, "x": 0.02, "xanchor": "left"},
        paper_bgcolor=ACTIVE_THEME["panel_bg"],
        plot_bgcolor=ACTIVE_THEME["panel_bg"],
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=ACTIVE_THEME["grid"], zeroline=False)
    return fig


def classify_portfolio(summary: dict[str, float]) -> tuple[str, str, str]:
    sharpe = summary["Sharpe Ratio"]
    drawdown = summary["Max Drawdown"]
    volatility = summary["Annual Volatility"]

    if sharpe >= 1 and drawdown > -0.2:
        status = "Dengeli"
        message = "Risk basina getiri uretimi ve maksimum dusus seviyesi kabul edilebilir aralikta."
        color = "#166534"
    elif sharpe >= 0.5 and drawdown > -0.3:
        status = "Izlenebilir"
        message = "Portfoy makul bir risk-getiri profili sunuyor; volatilite ve yogunlasma izlenmeli."
        color = "#92400e"
    else:
        status = "Riskli"
        message = "Risk basina getiri zayif; varlik dagilimi ve kayip toleransi yeniden degerlendirilmeli."
        color = "#991b1b"

    if volatility > 0.25:
        message = "Yillik volatilite yuksek; daha defansif varliklar veya daha dengeli dagilim degerlendirilebilir."
    if drawdown <= -0.35:
        message = "Gecmis maksimum dusus seviyesi yuksek; portfoy kayip toleransiyla uyumlu olmayabilir."

    return status, message, color


def decision_cards(
    summary: dict[str, float],
    current_sharpe: float,
    optimized_sharpe: float,
    current_volatility: float,
    min_volatility: float,
) -> None:
    status, message, color = classify_portfolio(summary)
    sharpe_gap = optimized_sharpe - current_sharpe
    risk_gap = current_volatility - min_volatility

    st.markdown(
        f"""
        <div class="decision-grid" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0 18px 0;">
            <div class="theme-card">
                <div class="theme-muted">Genel Durum</div>
                <div style="color:{color};font-size:1.45rem;font-weight:700;margin-top:6px;">{status}</div>
                <div class="theme-muted" style="margin-top:8px;">{message}</div>
            </div>
            <div class="theme-card">
                <div class="theme-muted">Optimizasyon Firsati</div>
                <div style="color:{ACTIVE_THEME["text"]};font-size:1.45rem;font-weight:700;margin-top:6px;">{format_signed_number(sharpe_gap)}</div>
                <div class="theme-muted" style="margin-top:8px;">Max Sharpe portfoyu ile Sharpe farki.</div>
            </div>
            <div class="theme-card">
                <div class="theme-muted">Risk Azaltma Alani</div>
                <div style="color:{ACTIVE_THEME["text"]};font-size:1.45rem;font-weight:700;margin-top:6px;">{format_delta(risk_gap)}</div>
                <div class="theme-muted" style="margin-top:8px;">Min risk portfoyune gore volatilite farki.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def preset_defaults(assets: list[str], preset_name: str) -> dict[str, float]:
    preset = PRESET_PORTFOLIOS[preset_name]
    if preset is None:
        return {asset: 100 / len(assets) for asset in assets}
    raw = {asset: preset.get(asset, 0.0) for asset in assets}
    total = sum(raw.values())
    if total <= 0:
        return {asset: 100 / len(assets) for asset in assets}
    return {asset: value / total * 100 for asset, value in raw.items()}


def default_lots_from_weights(defaults: dict[str, float], latest_prices: pd.Series) -> dict[str, int]:
    target_value = 10_000
    lots: dict[str, int] = {}
    for asset, price in latest_prices.items():
        target_weight = defaults.get(asset, 100 / len(latest_prices)) / 100
        lots[asset] = max(1, int(round((target_value * target_weight) / price))) if price > 0 else 1
    return lots


def build_lot_controls(
    assets: list[str],
    default_lots: dict[str, int],
    latest_prices: pd.Series,
    key_prefix: str,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    columns = st.columns(min(len(assets), 4))
    raw_lots = []

    for index, asset in enumerate(assets):
        with columns[index % len(columns)]:
            raw_lots.append(
                st.number_input(
                    asset,
                    min_value=0,
                    max_value=1_000_000,
                    value=int(default_lots.get(asset, 1)),
                    step=1,
                    key=f"{key_prefix}_lot_{asset}",
                    help=f"Son fiyat: {format_money(latest_prices.get(asset, np.nan))}",
                )
            )

    lots = pd.Series(raw_lots, index=assets, dtype=float)
    weights_series = lots_to_weights(lots, latest_prices)
    values = lots_to_position_values(lots, latest_prices)
    if np.isclose(values.sum(), 0):
        st.warning("Lot degerleri sifir. Hesaplama icin esit agirlik kullaniliyor.")
    return weights_series.values, lots, values


def metric_cards(summary: dict[str, float]) -> None:
    columns = st.columns(4)
    display_items = [
        ("Yillik Getiri", format_percent(summary["Annual Return"])),
        ("Yillik Risk", format_percent(summary["Annual Volatility"])),
        ("Sharpe Ratio", format_number(summary["Sharpe Ratio"])),
        ("En Buyuk Dusus", format_percent(summary["Max Drawdown"])),
    ]
    for column, (label, value) in zip(columns, display_items):
        column.metric(label, value)


def format_cell_value(value: object, formatter: str | callable | None, na_rep: str) -> str:
    if pd.isna(value):
        return na_rep
    if formatter is None:
        if isinstance(value, float):
            return f"{value:,.4f}"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)
    if isinstance(formatter, str):
        return formatter.format(value)
    return str(formatter(value))


def show_table(
    table: pd.DataFrame | pd.Series,
    formatters: dict[str, str | callable] | str | callable | None = None,
    na_rep: str = "",
    include_index: bool | None = None,
    max_height: int = 420,
) -> None:
    frame = table.to_frame() if isinstance(table, pd.Series) else table.copy()
    if include_index is None:
        include_index = not isinstance(frame.index, pd.RangeIndex)

    header_cells = []
    if include_index:
        index_name = frame.index.name or ""
        header_cells.append(f"<th>{escape(str(index_name))}</th>")
    header_cells.extend(f"<th>{escape(str(column))}</th>" for column in frame.columns)

    body_rows = []
    for index_value, row in frame.iterrows():
        cells = []
        if include_index:
            cells.append(f"<td>{escape(format_cell_value(index_value, None, na_rep))}</td>")
        for column, value in row.items():
            formatter = formatters.get(column) if isinstance(formatters, dict) else formatters
            cell_class = "numeric-cell" if isinstance(value, (int, float, np.integer, np.floating)) else ""
            cells.append(f'<td class="{cell_class}">{escape(format_cell_value(value, formatter, na_rep))}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    html = f"""
    <div class="data-table-wrap" style="max-height:{max_height}px;">
        <table class="data-table">
            <thead><tr>{''.join(header_cells)}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def asset_metric_display_table(table: pd.DataFrame) -> pd.DataFrame:
    renamed = table.rename(
        columns={
            "Annual Return": "Yillik Getiri",
            "Annual Volatility": "Yillik Risk",
            "Sharpe": "Sharpe",
            "Sortino": "Sortino",
            "Max Drawdown": "En Buyuk Dusus",
            "Beta": "Beta",
        }
    )
    return renamed


ASSET_METRIC_FORMATTERS = {
    "Yillik Getiri": "{:.2%}",
    "Yillik Risk": "{:.2%}",
    "Sharpe": "{:.2f}",
    "Sortino": "{:.2f}",
    "En Buyuk Dusus": "{:.2%}",
    "Beta": "{:.2f}",
}


def allocation_action_table(current_weights: pd.Series, target_weights: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Mevcut": current_weights,
            "Onerilen": target_weights,
        }
    )
    frame["Fark"] = frame["Onerilen"] - frame["Mevcut"]
    frame["Aksiyon"] = np.select(
        [frame["Fark"] > 0.03, frame["Fark"] < -0.03],
        ["Artir", "Azalt"],
        default="Koru",
    )
    return frame.sort_values("Fark", key=lambda series: series.abs(), ascending=False)


def sidebar_analysis_guide(
    summary: dict[str, float],
    assets: list[str],
    source: str,
    benchmark: str,
    analysis_profile_name: str,
) -> None:
    status, message, _ = classify_portfolio(summary)
    sidebar_step(
        5,
        "Analiz Rehberi",
        "Ilk bakista portfoy durumunu ve hangi sekmeye oncelik verecegini gosterir.",
    )
    st.metric("Portfoy durumu", status)
    st.caption(message)
    st.markdown(
        f"""
        <div class="sidebar-status">
            Varlik sayisi: <strong>{len(assets)}</strong><br>
            Veri kaynagi: <strong>{source}</strong><br>
            Benchmark: <strong>{benchmark}</strong><br>
            Profil: <strong>{analysis_profile_name}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if status == "Riskli":
        st.warning("Once Risk sekmesindeki risk katkisi ve maksimum dusus alanlarini incele.")
    elif status == "Izlenebilir":
        st.info("Ozet ve Optimizasyon sekmelerini birlikte kontrol et.")
    else:
        st.success("Portfoy dengeli gorunuyor; Analiz sekmesindeki stres testleriyle teyit et.")


def sidebar_step(number: int, title: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="sidebar-kicker">{number}. ADIM</div>
        <h3 style="margin-top:0;">{title}</h3>
        <div class="theme-muted" style="margin-top:-0.4rem;margin-bottom:0.75rem;">{note}</div>
        """,
        unsafe_allow_html=True,
    )


def build_ticker_input() -> str:
    sidebar_step(
        1,
        "Hisse Ekle",
        "Yahoo Finance sembolunu yaz, portfoye ekle ve listeyi altta takip et.",
    )

    if "portfolio_assets" not in st.session_state:
        st.session_state["portfolio_assets"] = list(DEFAULT_TICKERS)

    search_value = st.text_input(
        "Hisse ara veya ekle",
        value="",
        placeholder="Orn: AAPL",
        help="Tek sembol veya virgulle ayrilmis birden fazla Yahoo Finance sembolu ekleyebilirsin.",
        key="asset_search_input",
    )
    add_clicked = st.button("Portfoye ekle", type="primary", use_container_width=True)
    if add_clicked:
        new_assets = parse_tickers(search_value)
        if new_assets:
            st.session_state["portfolio_assets"] = list(
                dict.fromkeys([*st.session_state["portfolio_assets"], *new_assets])
            )
        else:
            st.warning("Eklemek icin gecerli bir hisse sembolu yaz.")

    st.markdown("#### Portfoy")
    for asset in list(st.session_state["portfolio_assets"]):
        asset_column, remove_column = st.columns([0.82, 0.18])
        with asset_column:
            st.markdown(f'<div class="portfolio-asset-label">{asset}</div>', unsafe_allow_html=True)
        with remove_column:
            if st.button("x", key=f"remove_asset_{asset}", help=f"{asset} portfoyden cikar"):
                st.session_state["portfolio_assets"] = [
                    current_asset for current_asset in st.session_state["portfolio_assets"] if current_asset != asset
                ]
                st.rerun()
    selected_text = ", ".join(st.session_state["portfolio_assets"]) or "Portfoy bos"
    st.markdown(
        f"""
        <div class="sidebar-status">
            Portfoydeki hisse sayisi: <strong>{len(st.session_state["portfolio_assets"])}</strong><br>
            <span class="theme-muted">{selected_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return ", ".join(st.session_state["portfolio_assets"])


st.title("Portfoy Analizi Dashboard")

with st.sidebar:
    raw_tickers = build_ticker_input()

    sidebar_step(
        2,
        "Veriyi Hazirla",
        "Fiyat serisi yalnizca canli kaynaktan yuklenir; sample veri dashboardda kullanilmaz.",
    )
    today = date.today()
    start = st.date_input("Baslangic tarihi", today - timedelta(days=365 * 4))
    end = st.date_input("Bitis tarihi", today)
    data_mode_label = "Canli veri"
    data_mode = "live"
    st.markdown(
        '<div class="sidebar-status"><strong>Canli veri aktif</strong><br>Yahoo Finance fiyat verisi, Finnhub sirket profili ve haber verisi.</div>',
        unsafe_allow_html=True,
    )

    sidebar_step(
        3,
        "Analizi Ayarla",
        "Yorumlama profilini ve varsayimlari risk-getiri hedefinle eslestir.",
    )
    analysis_profile_name = st.selectbox(
        "Yorumlama profili",
        list(ANALYSIS_PROFILES.keys()),
        help="Dashboard metriklerinin risk, getiri ve senaryo tarafinda hangi oncelikle yorumlanacagini belirler.",
    )
    analysis_profile = ANALYSIS_PROFILES[analysis_profile_name]
    st.caption(analysis_profile["description"])
    profile_key = analysis_profile_name.replace(" ", "_")

    with st.expander("Gelismis varsayimlar"):
        risk_free_rate = st.slider(
            "Risksiz faiz",
            0.0,
            0.12,
            analysis_profile["risk_free_rate"],
            0.005,
            key=f"{profile_key}_risk_free_rate",
            help="Portföy getirisinin risksiz getiri üzerinde ne kadar değer ürettiğini ölçmek için Sharpe ve Sortino hesaplarında kullanılır.",
        )
        confidence = st.slider(
            "VaR guven duzeyi",
            0.90,
            0.99,
            analysis_profile["confidence"],
            0.01,
            key=f"{profile_key}_confidence",
            help="VaR ve CVaR metriklerinde kayıp eşiğinin ne kadar temkinli hesaplanacağını belirler; değer yükseldikçe risk okuması daha muhafazakar olur.",
        )
        initial_value = st.number_input(
            "Portfoy degeri",
            1000,
            1_000_000,
            analysis_profile["initial_value"],
            step=1000,
            key=f"{profile_key}_initial_value",
            help="Simülasyon sekmesinde olası portföy değer aralıklarını para cinsinden göstermek için kullanılan başlangıç tutarıdır.",
        )
        transaction_cost_bps = st.slider(
            "Islem maliyeti (bps)",
            0,
            100,
            analysis_profile["transaction_cost_bps"],
            5,
            key=f"{profile_key}_transaction_cost_bps",
            help="Optimizasyon önerisindeki ağırlık değişimlerinin yaratacağı yaklaşık alım-satım maliyetini baz puan cinsinden hesaplar.",
        )

requested_tickers = parse_tickers(raw_tickers)
input_validation = validate_requested_inputs(requested_tickers, start, end)
for warning in input_validation.warnings:
    st.warning(warning)
for error in input_validation.errors:
    st.error(error)
if not input_validation.ok:
    st.stop()

result = load_prices(raw_tickers, start, end, data_mode)
if result.warning:
    st.warning(result.warning)

prices = result.prices
returns = calculate_returns(prices)

dataset_validation = validate_clean_dataset(prices, returns)
for warning in dataset_validation.warnings:
    st.warning(warning)
for error in dataset_validation.errors:
    st.error(error)
if not dataset_validation.ok:
    st.stop()

assets = list(prices.columns)
st.caption(f"Kaynak: {result.source} | Varliklar: {', '.join(assets)}")

with st.sidebar:
    sidebar_step(
        4,
        "Pozisyonlari Gir",
        "Lotlari gir; dashboard pozisyon degerini ve agirligi otomatik hesaplar.",
    )
    latest_prices = prices.iloc[-1]
    benchmark = st.selectbox(
        "Karsilastirma varligi",
        assets,
        index=0,
        help="Portfoy performansini bu varlikla karsilastirir.",
    )
    preset_name = st.selectbox(
        "Baslangic dagilimi",
        list(PRESET_PORTFOLIOS.keys()),
        help="Lot kutulari icin ilk dagilim onerisi verir; son karar lot girisleriyle belirlenir.",
    )
    with st.expander("Lotlari duzenle", expanded=True):
        st.caption("Ornek: AAPL icin 10 lot girersen, 10 x son fiyat portfoy degerine eklenir.")
        weights, lots, position_values = build_lot_controls(
            assets,
            default_lots_from_weights(preset_defaults(assets, preset_name), latest_prices),
            latest_prices,
            preset_name.replace(" ", "_"),
        )
        show_table(
            pd.DataFrame(
                {
                    "Varlik": assets,
                    "Lot": lots,
                    "Fiyat": latest_prices.values,
                    "Agirlik": pd.Series(weights, index=assets).values,
                }
            ),
            {
                "Lot": "{:.0f}",
                "Fiyat": "${:,.2f}",
                "Agirlik": "{:.2%}",
            },
            include_index=False,
            max_height=260,
        )

for warning in concentration_warnings(weights, assets):
    st.warning(warning)

portfolio_daily_returns = portfolio_returns(returns, weights)
summary = portfolio_metric_summary(portfolio_daily_returns, risk_free_rate, confidence)

current_weights = pd.Series(weights, index=assets)
max_sharpe_weights = optimize_portfolio(returns, "max_sharpe", risk_free_rate)
min_vol_weights = optimize_portfolio(returns, "min_volatility", risk_free_rate)
expected_returns = returns.mean() * 252
covariance = returns.cov() * 252
current_return, current_volatility, current_sharpe = portfolio_performance(
    current_weights.values,
    expected_returns,
    covariance,
    risk_free_rate,
)
max_sharpe_return, max_sharpe_volatility, optimized_sharpe = portfolio_performance(
    max_sharpe_weights.values,
    expected_returns,
    covariance,
    risk_free_rate,
)
min_vol_return, min_volatility, min_vol_sharpe = portfolio_performance(
    min_vol_weights.values,
    expected_returns,
    covariance,
    risk_free_rate,
)

metric_cards(summary)
decision_cards(summary, current_sharpe, optimized_sharpe, current_volatility, min_volatility)

st.info(
    f"{data_mode_label} aktif. Baslangic dagilimi: {preset_name}. "
    f"Benchmark: {benchmark}. Pozisyon agirliklari lot ve son fiyat uzerinden hesaplandi."
)

with st.sidebar:
    sidebar_analysis_guide(summary, assets, result.source, benchmark, analysis_profile_name)

tabs = st.tabs(["Ozet", "Risk", "Analiz", "Optimizasyon", "Simulasyon", "Sirket Verisi", "Veri"])

with tabs[0]:
    left, right = st.columns([1.45, 1])

    with left:
        growth = cumulative_growth(returns[assets], initial_value=100)
        growth["Portfoy"] = cumulative_growth(portfolio_daily_returns, initial_value=100)
        fig = px.line(
            growth,
            labels={"value": "100 birimlik yatirim", "index": "Tarih", "variable": "Varlik"},
            title="Getiri Gelisimi",
            color_discrete_sequence=[
                PRIMARY_COLOR,
                "#0f766e",
                "#f59e0b",
                WARNING_COLOR,
                "#7c3aed",
                "#475569",
            ],
        )
        st.plotly_chart(clean_chart(fig, height=430), use_container_width=True)

    with right:
        weights_frame = pd.DataFrame({"Asset": assets, "Weight": weights})
        fig = px.bar(
            weights_frame,
            x="Weight",
            y="Asset",
            orientation="h",
            title="Portfoy Dagilimi",
            labels={"Weight": "Agirlik", "Asset": "Varlik"},
            text=weights_frame["Weight"].map(lambda value: f"{value:.0%}"),
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        fig.update_xaxes(tickformat=".0%", range=[0, max(weights.max() * 1.18, 0.35)])
        fig.update_traces(textposition="outside", hovertemplate="%{y}: %{x:.2%}<extra></extra>")
        st.plotly_chart(clean_chart(fig, height=430), use_container_width=True)

    st.subheader("Varlik Bazli Ozet")
    table = asset_metric_table(returns, benchmark=benchmark, risk_free_rate=risk_free_rate)
    show_table(asset_metric_display_table(table), ASSET_METRIC_FORMATTERS, na_rep="n/a")

    summary_export = pd.DataFrame(summary.items(), columns=["Metrik", "Deger"])
    action_export = allocation_action_table(current_weights, max_sharpe_weights)
    html_report = build_html_report(
        "Portfoy Analizi Raporu",
        result.source,
        assets,
        summary,
        current_weights,
        action_export,
    )
    export_left, export_mid, export_right = st.columns(3)
    export_left.download_button(
        "Ozet CSV",
        data=dataframe_to_csv_bytes(summary_export),
        file_name="portfolio_summary.csv",
        mime="text/csv",
    )
    export_mid.download_button(
        "Aksiyon CSV",
        data=dataframe_to_csv_bytes(action_export),
        file_name="portfolio_actions.csv",
        mime="text/csv",
    )
    export_right.download_button(
        "HTML Rapor",
        data=html_report.encode("utf-8"),
        file_name="portfolio_report.html",
        mime="text/html",
    )

with tabs[1]:
    left, right = st.columns(2)

    with left:
        st.caption("Korelasyon, varliklarin ayni yonde hareket etme egilimini gosterir. 1'e yakin degerler birlikte hareketi, 0'a yakin degerler daha bagimsiz davranisi ifade eder.")
        corr = returns.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Korelasyon",
        )
        st.plotly_chart(clean_chart(fig, height=420), use_container_width=True)

    with right:
        st.caption("63 gunluk risk, portfoyun son donemdeki oynakligini yilliklandirilmis olarak izler. Yukselis, portfoy dalgalanmasinin arttigina isaret eder.")
        rolling_vol = portfolio_daily_returns.rolling(63).std() * np.sqrt(252)
        fig = px.line(
            rolling_vol,
            labels={"value": "Yillik risk", "index": "Tarih"},
            title="63 Gunluk Risk",
            color_discrete_sequence=[WARNING_COLOR],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(clean_chart(fig, height=420), use_container_width=True)

    st.subheader("Risk Ozeti")
    st.caption("Bu tablo portfoyun getiri, oynaklik, risk basina getiri ve asagi yonlu kayip metriklerini birlikte okur.")
    risk_frame = pd.DataFrame(
        [
            ("Yillik getiri", format_percent(summary["Annual Return"])),
            ("Yillik risk", format_percent(summary["Annual Volatility"])),
            ("Sharpe ratio", format_number(summary["Sharpe Ratio"])),
            ("Sortino ratio", format_number(summary["Sortino Ratio"])),
            ("En buyuk dusus", format_percent(summary["Max Drawdown"])),
            (f"VaR {int(confidence * 100)}%", format_percent(summary[f"VaR {int(confidence * 100)}%"])),
            (f"CVaR {int(confidence * 100)}%", format_percent(summary[f"CVaR {int(confidence * 100)}%"])),
        ],
        columns=["Metrik", "Deger"],
    )
    show_table(risk_frame)
    risk_explanations = pd.DataFrame(
        [
            ("Yillik getiri", "Portfoyun gecmis veriye gore yilliklandirilmis getiri oranidir."),
            ("Yillik risk", "Gunluk getirilerin yilliklandirilmis standart sapmasidir; oynakligi olcer."),
            ("Sharpe ratio", "Risksiz faiz uzerindeki getirinin toplam riske oranidir."),
            ("Sortino ratio", "Sadece asagi yonlu oynakligi dikkate alan risk basina getiri metriğidir."),
            ("En buyuk dusus", "Gecmiste zirveden dibe gorulen en sert portfoy kaybidir."),
            ("VaR", "Secilen guven duzeyinde beklenen gunluk kayip esigidir."),
            ("CVaR", "VaR esiginin asildigi kotu gunlerde beklenen ortalama kaybi gosterir."),
        ],
        columns=["Metrik", "Yorum"],
    )
    show_table(risk_explanations)

    st.subheader("Benchmark Karsilastirmasi")
    benchmark_frame = benchmark_comparison(
        portfolio_daily_returns,
        returns[benchmark],
        risk_free_rate=risk_free_rate,
    )
    show_table(
        benchmark_frame,
        {
            "Yillik Getiri": "{:.2%}",
            "Yillik Risk": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max Dusus": "{:.2%}",
            "Tracking Error": "{:.2%}",
            "Information Ratio": "{:.2f}",
        },
        na_rep="",
    )

    left_decision, right_decision = st.columns(2)
    with left_decision:
        st.subheader("Yogunlasma")
        st.caption("Yogunlasma, portfoyun az sayida varliga bagimli olup olmadigini gosterir. Efektif varlik sayisi dustukce cesitlendirme zayiflar.")
        concentration_frame = concentration_summary(weights, assets)
        show_table(
            concentration_frame,
            {
                "Deger": lambda value: f"{value:.2%}" if isinstance(value, float) and value <= 1 else f"{value:.2f}" if isinstance(value, float) else value
            },
        )

    with right_decision:
        st.subheader("Risk Katkisi")
        st.caption("Risk katkisi, toplam portfoy oynakliginin hangi varliklardan kaynaklandigini gosterir. Agirligi dusuk ama oynakligi yuksek varliklar riskte daha buyuk pay alabilir.")
        risk_contribution_frame = risk_contribution(returns, weights)
        fig = px.bar(
            risk_contribution_frame.reset_index(),
            x="Risk Contribution",
            y="Asset",
            orientation="h",
            title="Varlik Bazli Risk Katkisi",
            labels={"Risk Contribution": "Risk katkisi", "Asset": "Varlik"},
            text=risk_contribution_frame["Risk Contribution"].map(lambda value: f"{value:.0%}"),
            color_discrete_sequence=[WARNING_COLOR],
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_traces(textposition="outside", hovertemplate="%{y}: %{x:.2%}<extra></extra>")
        st.plotly_chart(clean_chart(fig, height=360), use_container_width=True)

    st.subheader("Metrik Sozlugu")
    metric_dictionary = pd.DataFrame(
        [
            ("Sharpe", "Risk basina uretilen ek getiriyi gosterir; yuksek olmasi tercih edilir."),
            ("Yillik Risk", "Getirilerin yillik oynakligini gosterir; yuksek deger daha dalgali portfoy demektir."),
            ("En Buyuk Dusus", "Gecmiste zirveden dibe gorulen en sert kaybi gosterir."),
            ("VaR", "Secilen guven duzeyinde gunluk kaybin asabilecegi esik degeri gosterir."),
            ("Beta", "Benchmark hareketlerine duyarliligi gosterir; 1 civari benchmark ile benzer hareket demektir."),
        ],
        columns=["Metrik", "Anlam"],
    )
    show_table(metric_dictionary)

with tabs[2]:
    st.subheader("Rebalancing Karsilastirmasi")
    st.caption("Rebalancing analizi, portfoyun belirli araliklarla hedef dagilima geri cekilmesi durumunda getiri ve risk profilinin nasil degistigini gosterir.")
    rebalance_frame = rebalancing_comparison(returns, weights)
    show_table(
        rebalance_frame,
        {
            "Yillik Getiri": "{:.2%}",
            "Yillik Risk": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Max Dusus": "{:.2%}",
        },
    )

    st.subheader("Rolling Metrikler")
    st.caption("Rolling metrikler, performansin zaman icindeki istikrarini izlemek icin kullanilir; tek bir donem ortalamasina bagli kalmadan trendi gosterir.")
    rolling_frame = rolling_metrics(portfolio_daily_returns, window=min(26, max(5, len(portfolio_daily_returns) // 3)), risk_free_rate=risk_free_rate)
    left_roll, right_roll = st.columns(2)
    with left_roll:
        fig = px.line(
            rolling_frame["Rolling Return"],
            title="Rolling Getiri",
            labels={"value": "Yillik getiri", "index": "Tarih"},
            color_discrete_sequence=[ACCENT_COLOR],
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(clean_chart(fig, height=360), use_container_width=True)
    with right_roll:
        fig = px.line(
            rolling_frame["Rolling Sharpe"],
            title="Rolling Sharpe",
            labels={"value": "Sharpe", "index": "Tarih"},
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        st.plotly_chart(clean_chart(fig, height=360), use_container_width=True)

    st.subheader("Stres Testleri")
    st.caption("Stres testleri, secili tarih araliklarinda portfoyun kayip ve toparlanma davranisini inceler. Bu bolum olasi kriz hassasiyetini okumaya yardimci olur.")
    stress_frame = stress_test_summary(portfolio_daily_returns)
    show_table(
        stress_frame,
        {
            "Getiri": "{:.2%}",
            "Max Dusus": "{:.2%}",
            "Gozlem": "{:.0f}",
        },
        na_rep="Veri yok",
    )

with tabs[3]:
    portfolios = random_portfolios(returns, n_portfolios=3500, risk_free_rate=risk_free_rate)

    opt_left, opt_mid, opt_right = st.columns(3)
    opt_left.metric("Mevcut Sharpe", format_number(current_sharpe))
    opt_left.caption("Mevcut lot dagiliminin risk basina getiri skoru.")
    opt_mid.metric("Max Sharpe", format_number(optimized_sharpe), delta=format_signed_number(optimized_sharpe - current_sharpe))
    opt_mid.caption("Ayni varliklar icinde teorik olarak en yuksek Sharpe oranina yaklasan dagilim.")
    opt_right.metric(
        "Min Risk",
        format_percent(min_volatility),
        delta=format_delta(min_volatility - current_volatility),
        delta_color="inverse",
    )
    opt_right.caption("Beklenen volatiliteyi minimize eden teorik dagilim.")

    fig = px.scatter(
        portfolios,
        x="Annual Volatility",
        y="Annual Return",
        color="Sharpe",
        color_continuous_scale="Blues",
        title="Risk ve Getiri Dengesi",
        labels={
            "Annual Volatility": "Yillik risk",
            "Annual Return": "Yillik getiri",
        },
    )

    for label, optimized_weights in [
        ("Mevcut", current_weights),
        ("Max Sharpe", max_sharpe_weights),
        ("Min Risk", min_vol_weights),
    ]:
        point_return, volatility, sharpe = portfolio_performance(
            optimized_weights.values,
            expected_returns,
            covariance,
            risk_free_rate,
        )
        marker_color = PRIMARY_COLOR if label == "Mevcut" else ACCENT_COLOR
        fig.add_trace(
            go.Scatter(
                x=[volatility],
                y=[point_return],
                mode="markers+text",
                marker={"size": 14, "symbol": "diamond", "color": marker_color},
                text=[label],
                textposition="top center",
                name=f"{label} ({sharpe:.2f})",
            )
        )

    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(clean_chart(fig, height=470), use_container_width=True)

    st.subheader("Onerilen Degisiklik")
    action_frame = allocation_action_table(current_weights, max_sharpe_weights)
    show_table(
        action_frame,
        {
            "Mevcut": "{:.2%}",
            "Onerilen": "{:.2%}",
            "Fark": "{:+.2%}",
        },
    )

    cost_impact = transaction_cost_impact(current_weights, max_sharpe_weights, transaction_cost_bps)
    cost_left, cost_right = st.columns(2)
    cost_left.metric("Tahmini Turnover", format_percent(cost_impact["Turnover"]))
    cost_right.metric("Tek Seferlik Maliyet", format_percent(cost_impact["Tahmini Maliyet"]))

    st.subheader("Portfoy Karsilastirmasi")
    performance_comparison = pd.DataFrame(
        {
            "Yillik Getiri": [current_return, max_sharpe_return, min_vol_return],
            "Yillik Risk": [current_volatility, max_sharpe_volatility, min_volatility],
            "Sharpe": [current_sharpe, optimized_sharpe, min_vol_sharpe],
        },
        index=["Mevcut", "Max Sharpe", "Min Risk"],
    )
    show_table(
        performance_comparison,
        {
            "Yillik Getiri": "{:.2%}",
            "Yillik Risk": "{:.2%}",
            "Sharpe": "{:.2f}",
        },
    )

    st.subheader("Agirlik Karsilastirmasi")
    comparison = pd.DataFrame(
        {
            "Mevcut": current_weights,
            "Max Sharpe": max_sharpe_weights,
            "Min Risk": min_vol_weights,
        }
    )
    show_table(comparison, "{:.2%}")

with tabs[4]:
    horizon_days = st.slider("Simulasyon suresi", 63, 756, 252, 21)
    n_simulations = st.slider("Simulasyon sayisi", 100, 2000, 500, 100)
    paths = monte_carlo_paths(
        portfolio_daily_returns,
        initial_value=initial_value,
        horizon_days=horizon_days,
        n_simulations=n_simulations,
    )
    percentiles = pd.DataFrame(
        {
            "P05": paths.quantile(0.05, axis=1),
            "P50": paths.quantile(0.50, axis=1),
            "P95": paths.quantile(0.95, axis=1),
        }
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles["P95"], line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["P05"],
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.16)",
            line={"width": 0},
            name="5%-95% aralik",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles["P50"],
            mode="lines",
            line={"color": PRIMARY_COLOR, "width": 3},
            name="Medyan",
        )
    )
    fig.update_layout(title="Olasilik Araligi", xaxis_title="Islem gunu", yaxis_title="Portfoy degeri")
    st.plotly_chart(clean_chart(fig, height=450), use_container_width=True)

    terminal_summary = terminal_value_summary(paths)
    scenario_left, scenario_mid, scenario_right = st.columns(3)
    scenario_left.metric("Dusuk Senaryo", format_money(terminal_summary["P05"]))
    scenario_mid.metric("Orta Senaryo", format_money(terminal_summary["Median"]))
    scenario_right.metric("Guclu Senaryo", format_money(terminal_summary["P95"]))

    terminal_table = terminal_summary.rename(
        {
            "P05": "Dusuk senaryo",
            "P25": "Zayif senaryo",
            "Median": "Orta senaryo",
            "P75": "Iyi senaryo",
            "P95": "Guclu senaryo",
        }
    )
    show_table(terminal_table.to_frame("Donem sonu degeri"), format_money)

with tabs[5]:
    st.subheader("Finnhub Sirket Verisi")
    if not has_finnhub_key():
        st.warning("Finnhub sirket verisi icin FINNHUB_API_KEY gerekli.")
    else:
        snapshot = fetch_finnhub_snapshot(assets, news_per_ticker=2)
        if snapshot.warning:
            st.warning(snapshot.warning)
        if snapshot.profiles:
            profile_frame = pd.DataFrame([profile.__dict__ for profile in snapshot.profiles])
            profile_frame = profile_frame.rename(
                columns={
                    "ticker": "Varlik",
                    "name": "Sirket",
                    "exchange": "Borsa",
                    "industry": "Sektor",
                    "country": "Ulke",
                    "market_cap": "Piyasa Degeri",
                }
            )
            show_table(profile_frame)
        if snapshot.news:
            news_frame = pd.DataFrame([item.__dict__ for item in snapshot.news])
            news_frame = news_frame.rename(
                columns={
                    "ticker": "Varlik",
                    "headline": "Baslik",
                    "source": "Kaynak",
                    "url": "URL",
                    "published_at": "Tarih",
                }
            )
            show_table(news_frame, max_height=320)
        if not snapshot.profiles and not snapshot.news and not snapshot.warning:
            st.info("Finnhub veri donmedi.")

with tabs[6]:
    st.subheader("Veri Ozeti")
    data_summary = pd.DataFrame(
        [
            ("Kaynak", result.source),
            ("Varlik sayisi", f"{len(assets)}"),
            ("Ilk tarih", prices.index.min().strftime("%Y-%m-%d")),
            ("Son tarih", prices.index.max().strftime("%Y-%m-%d")),
            ("Gozlem sayisi", f"{len(prices):,}"),
        ],
        columns=["Alan", "Deger"],
    )
    show_table(data_summary)

    st.subheader("Fiyat Verisi")
    show_table(prices.tail(250), max_height=520)
    st.subheader("Gunluk Getiri")
    show_table(returns.tail(250), max_height=520)
