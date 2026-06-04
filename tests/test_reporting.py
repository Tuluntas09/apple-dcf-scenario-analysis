import pandas as pd

from src.reporting import build_html_report, dataframe_to_csv_bytes


def test_dataframe_to_csv_bytes_returns_utf8_bytes():
    data = dataframe_to_csv_bytes(pd.DataFrame({"A": [1]}))

    assert isinstance(data, bytes)
    assert b"A" in data


def test_build_html_report_contains_sections():
    html = build_html_report(
        "Rapor",
        "Sample veri",
        ["SPY", "QQQ"],
        {"Annual Return": 0.1, "Sharpe Ratio": 1.2},
        pd.Series({"SPY": 0.5, "QQQ": 0.5}),
        pd.DataFrame(
            {
                "Mevcut": [0.5],
                "Onerilen": [0.6],
                "Fark": [0.1],
                "Aksiyon": ["Artir"],
            },
            index=["SPY"],
        ),
    )

    assert "Ozet Metrikler" in html
    assert "Onerilen Degisiklik" in html
    assert "SPY" in html
