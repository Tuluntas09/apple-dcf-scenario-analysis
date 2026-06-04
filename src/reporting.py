from __future__ import annotations

from html import escape

import pandas as pd


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=True).encode("utf-8")


def build_html_report(
    title: str,
    source: str,
    assets: list[str],
    summary: dict[str, float],
    weights: pd.Series,
    action_table: pd.DataFrame,
) -> str:
    summary_rows = "\n".join(
        f"<tr><td>{escape(str(key))}</td><td>{value:.4f}</td></tr>"
        for key, value in summary.items()
        if isinstance(value, int | float)
    )
    weight_rows = "\n".join(
        f"<tr><td>{escape(asset)}</td><td>{weight:.2%}</td></tr>"
        for asset, weight in weights.items()
    )
    action_rows = "\n".join(
        f"<tr><td>{escape(str(index))}</td><td>{row['Mevcut']:.2%}</td><td>{row['Onerilen']:.2%}</td><td>{row['Fark']:+.2%}</td><td>{escape(str(row['Aksiyon']))}</td></tr>"
        for index, row in action_table.iterrows()
    )
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #111827; margin: 32px; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #64748b; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="muted">Kaynak: {escape(source)} | Varliklar: {escape(", ".join(assets))}</p>
  <h2>Ozet Metrikler</h2>
  <table><tbody>{summary_rows}</tbody></table>
  <h2>Portfoy Agirliklari</h2>
  <table><tbody>{weight_rows}</tbody></table>
  <h2>Onerilen Degisiklik</h2>
  <table>
    <thead><tr><th>Varlik</th><th>Mevcut</th><th>Onerilen</th><th>Fark</th><th>Aksiyon</th></tr></thead>
    <tbody>{action_rows}</tbody>
  </table>
</body>
</html>"""
