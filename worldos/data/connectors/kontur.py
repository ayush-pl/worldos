"""Kontur Population connector: gridded population on H3 hexagons (CC BY 4.0).

Kontur Population (2023-11-01 release) estimates population on H3 resolution-8 hexagons by combining GHSL,
Facebook/Meta HRSL, Microsoft and Copernicus building footprints, and OpenStreetMap. Each national GeoPackage
(SQLite) is downloaded to a temporary file, read with the standard-library sqlite3 module, aggregated to the
country's model H3 resolution, and only the aggregate is kept.

Citation: Kontur Population Dataset, Kontur Inc., via the Humanitarian Data Exchange. Licence: CC BY 4.0.
"""

from __future__ import annotations

import csv
import gzip
import shutil
import sqlite3
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import h3

from worldos.data.connectors.common import DEFAULT_COUNTRIES, REAL_DATA, write_manifest

URL = "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/kontur_population_{iso2}_20231101.gpkg.gz"
LICENSE = "CC BY 4.0"
ISO2 = {"IND": "IN", "USA": "US", "DEU": "DE", "JPN": "JP", "BRA": "BR", "NGA": "NG"}


def _model_resolution(iso3: str) -> int:
    from shapely.geometry import shape

    from worldos.countries.geography import choose_resolution
    from worldos.countries.sources import naturalearth

    country = next(f for f in naturalearth(iso3)["features"] if f["properties"]["level"] == "country")
    g = shape(country["geometry"])
    import math

    return choose_resolution(g.area * 111.0 * 111.0 * math.cos(math.radians(g.centroid.y)))


def fetch_kontur(countries: list[str] | None = None, out_dir: Path | None = None, iso2: dict | None = None) -> Path:
    countries = countries or DEFAULT_COUNTRIES
    iso2 = {**ISO2, **(iso2 or {})}
    out = out_dir or REAL_DATA / "kontur"
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for iso3 in countries:
        url = URL.format(iso2=iso2[iso3])
        res = _model_resolution(iso3)
        with tempfile.TemporaryDirectory() as tmp:
            gpkg = Path(tmp) / "pop.gpkg"
            with urllib.request.urlopen(url, timeout=1800) as r, gzip.GzipFile(fileobj=r) as gz, open(gpkg, "wb") as fh:
                shutil.copyfileobj(gz, fh, length=1 << 22)
            con = sqlite3.connect(gpkg)
            table = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchone()[0]
            agg: dict[str, float] = defaultdict(float)
            for cell, pop in con.execute(f'SELECT h3, population FROM "{table}"'):
                agg[h3.cell_to_parent(cell, res)] += float(pop)
            con.close()
        name = f"{iso3}_h3r{res}.csv"
        with open(out / name, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["h3", "population"])
            for cell, pop in sorted(agg.items()):
                w.writerow([cell, round(pop, 1)])
        entries.append({"file": name, "url": url, "rows": len(agg),
                        "filter": f"H3 resolution 8 aggregated to resolution {res}"})
    write_manifest(out, "Kontur Population 2023-11-01 (H3 r8), via HDX", LICENSE, entries,
                   notes="Population counts per H3 cell; aggregated by WORLD-OS to the model resolution.")
    return out
