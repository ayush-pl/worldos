"""Regional statistics and a self-contained SVG population map (no external dependencies)."""

from __future__ import annotations

import html
import math

import h3
import numpy as np
import pandas as pd

from worldos.population.tables import EMPLOYED, UNEMPLOYED


def regional_summary(world) -> pd.DataFrame:
    """Simulated population, unemployment and income by admin-1 region at the current tick."""
    p, h, lay = world.persons, world.households, world.layout
    alive = p.alive
    region = h.region[p.hh[alive]]
    w = p.weight[alive]
    st = p.status[alive]
    n = len(lay.region_ids)
    pop = np.bincount(region, weights=w, minlength=n)
    emp = np.bincount(region[st == EMPLOYED], weights=w[st == EMPLOYED], minlength=n)
    une = np.bincount(region[st == UNEMPLOYED], weights=w[st == UNEMPLOYED], minlength=n)
    hh = np.flatnonzero(h.alive & (h.n_members > 0))
    inc = np.bincount(h.region[hh], weights=h.income_last[hh], minlength=n)
    members = np.bincount(h.region[hh], weights=h.n_members[hh] * h.weight[hh], minlength=n)
    return pd.DataFrame({
        "region_id": lay.region_ids,
        "region": [world.geo[r].name for r in lay.region_ids],
        "population": pop,
        "population_share": pop / pop.sum(),
        "unemployment_rate": np.divide(une, emp + une, out=np.full(n, np.nan), where=(emp + une) > 0),
        "income_per_person_month": np.divide(inc, members, out=np.full(n, np.nan), where=members > 0),
    }).sort_values("population", ascending=False).reset_index(drop=True)


def population_map_svg(world, width: int = 900) -> str:
    """Choropleth of gridded population density on H3 cells (log scale), with the largest cities marked."""
    lay = world.layout
    if lay.cell_ids is None:
        return "<p>No gridded population layer for this world.</p>"
    cells, pop = lay.cell_ids, lay.cell_weight
    bounds = [h3.cell_to_boundary(c) for c in cells]
    lats = np.array([pt[0] for b in bounds for pt in b])
    lngs = np.array([pt[1] for b in bounds for pt in b])
    lat0 = float(np.mean(lats))
    kx = math.cos(math.radians(lat0))
    x0, x1, y0, y1 = lngs.min() * kx, lngs.max() * kx, lats.min(), lats.max()
    scale = width / (x1 - x0)
    height = int((y1 - y0) * scale) + 1
    density = pop / np.array([h3.cell_area(c, "km^2") for c in cells])
    logd = np.log10(np.maximum(density, 0.1))
    lo, hi = np.percentile(logd, 2), np.percentile(logd, 99.5)
    t = np.clip((logd - lo) / max(hi - lo, 1e-9), 0, 1)
    polys = []
    for b, ti in zip(bounds, t):
        pts = " ".join(f"{(lng * kx - x0) * scale:.1f},{(y1 - lat) * scale:.1f}" for lat, lng in b)
        light = 92 - 62 * ti  # light -> dark blue
        polys.append(f'<polygon points="{pts}" fill="hsl(212 70% {light:.0f}%)"/>')
    marks = []
    for cid in lay.city_ids[:10]:
        city = world.geo[cid]
        if city.attributes.get("aggregate"):
            continue
        lat, lng = city.centroid
        x, y = (lng * kx - x0) * scale, (y1 - lat) * scale
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="city"/>'
                     f'<text x="{x + 5:.1f}" y="{y - 4:.1f}" class="label">{html.escape(city.name)}</text>')
    legend = (f"Population per km² (log scale): light ≈ {10 ** lo:,.0f}, dark ≈ {10 ** hi:,.0f}. "
              f"{len(cells):,} populated H3 cells. Source: Kontur Population 2023 (CC BY 4.0).")
    return (f'<figure><svg viewBox="0 0 {width} {height}" role="img" aria-label="population density map">'
            f'<style>.city{{fill:#d9480f;stroke:#fff;stroke-width:1}}.label{{font:11px system-ui,sans-serif;'
            f'fill:currentColor;paint-order:stroke;stroke:var(--card,#fff);stroke-width:3px}}polygon{{stroke:none}}</style>'
            f'{"".join(polys)}{"".join(marks)}</svg><figcaption>{html.escape(legend)}</figcaption></figure>')
