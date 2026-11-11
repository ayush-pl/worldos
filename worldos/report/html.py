"""Self-contained HTML dashboard for a single run (inline SVG; works offline).

This is the Alpha stand-in for the Command Centre (docs/03, UI layer); the
React/CesiumJS application is a later milestone.
"""

from __future__ import annotations

import html
import math

import pandas as pd

PANELS = [
    ("unemployment_rate", "Unemployment rate", "pct"),
    ("inflation_yoy", "CPI inflation, year on year", "pct"),
    ("gdp_real", "Real output (goods / month)", "num"),
    ("mean_wage", "Mean monthly wage (SYN$)", "num"),
    ("policy_rate", "Policy rate", "pct"),
    ("gov_debt_to_gdp", "Public debt / annual GDP", "pct"),
    ("poverty_rate", "At-risk-of-poverty rate", "pct"),
    ("gini_income", "Gini, equivalised disposable income", "num"),
    ("population", "Population (represented persons)", "num"),
    ("bank_capital_ratio", "Bank equity / loans", "pct"),
]


def _fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.3g}"


def _chart(series: pd.Series, dates: list[str], kind: str, w: int = 320, h: int = 120) -> str:
    s = series.astype(float)
    valid = s.dropna()
    if len(valid) < 2:
        return '<p class="muted">insufficient data</p>'
    lo, hi = float(valid.min()), float(valid.max())
    if hi - lo < 1e-12:
        hi, lo = hi + 1e-9, lo - 1e-9
    pad = 4
    n = len(s)
    pts = []
    for i, v in enumerate(s):
        if math.isnan(v):
            continue
        x = pad + (w - 2 * pad) * i / max(n - 1, 1)
        y = pad + (h - 2 * pad) * (1 - (v - lo) / (hi - lo))
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="time series">'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" class="axis"/>'
            f'<polyline points="{" ".join(pts)}" class="line"/></svg>'
            f'<div class="range"><span>{html.escape(dates[0])}</span>'
            f'<span>min {_fmt(lo, kind)} · max {_fmt(hi, kind)}</span><span>{html.escape(dates[-1])}</span></div>')


def render_dashboard(metrics: pd.DataFrame, title: str, subtitle: str = "") -> str:
    dates = metrics["date"].astype(str).tolist()
    cards = []
    for col, label, kind in PANELS:
        if col not in metrics:
            continue
        last = metrics[col].dropna()
        latest = _fmt(float(last.iloc[-1]), kind) if len(last) else "n/a"
        cards.append(f'<section class="card"><h2>{html.escape(label)}</h2><div class="kpi">{latest}</div>'
                     f'{_chart(metrics[col], dates, kind)}</section>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#2f6db5; --axis:#d0d0cb; --warn:#8a5a00; --warnbg:#fff4dc; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141413; --fg:#ecece8; --muted:#a3a39d; --card:#1f1f1d; --line:#7fb0ff; --axis:#3a3a37; --warn:#f3c66b; --warnbg:#2c2414; }} }}
:root[data-theme="dark"] {{ --bg:#141413; --fg:#ecece8; --muted:#a3a39d; --card:#1f1f1d; --line:#7fb0ff; --axis:#3a3a37; --warn:#f3c66b; --warnbg:#2c2414; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} .muted {{ color:var(--muted); }}
.notice {{ background:var(--warnbg); color:var(--warn); padding:10px 12px; border-radius:8px; margin:16px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
.card {{ background:var(--card); border-radius:10px; padding:12px 14px; }}
.card h2 {{ font-size:13px; font-weight:600; color:var(--muted); margin:0; }}
.kpi {{ font-size:24px; font-weight:650; margin:2px 0 6px; font-variant-numeric:tabular-nums; }}
svg {{ width:100%; height:auto; display:block; }} .line {{ fill:none; stroke:var(--line); stroke-width:1.6; }}
.axis {{ stroke:var(--axis); }} .range {{ display:flex; justify-content:space-between; font-size:11px; color:var(--muted); gap:6px; flex-wrap:wrap; }}
</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="muted">{html.escape(subtitle)}</p>
<p class="notice">Synthetic country with assumed parameters. A model experiment, not a forecast or a description of any real economy.</p>
<div class="grid">{''.join(cards)}</div>
</main></body></html>"""
