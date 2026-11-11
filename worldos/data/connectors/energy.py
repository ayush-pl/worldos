"""Our World in Data energy dataset connector (primary energy by source; CC BY 4.0).

Underlying sources: Energy Institute Statistical Review of World Energy and Ember, compiled by OWID.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

from worldos.data.connectors.common import DEFAULT_COUNTRIES, REAL_DATA, write_manifest

URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
COLUMNS = ["iso_code", "country", "year", "primary_energy_consumption", "coal_consumption", "oil_consumption",
           "gas_consumption", "nuclear_consumption", "hydro_consumption", "wind_consumption", "solar_consumption",
           "biofuel_consumption", "other_renewable_consumption"]


def fetch_energy(countries: list[str] | None = None, min_year: int = 1990, out_dir: Path | None = None) -> Path:
    countries = set(countries or DEFAULT_COUNTRIES)
    out = out_dir or REAL_DATA / "energy"
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=300) as r:
        rows = [row for row in csv.DictReader(io.StringIO(r.read().decode("utf-8")))
                if (row["iso_code"] in countries or row["country"] == "World") and int(row["year"]) >= min_year]
    with open(out / "energy_owid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows([[r.get(c, "") for c in COLUMNS] for r in rows])
    write_manifest(out, "Our World in Data energy dataset (Energy Institute Statistical Review, Ember)", "CC BY 4.0",
                   [{"file": "energy_owid.csv", "url": URL, "rows": len(rows),
                     "filter": f"{sorted(countries)} + World, year >= {min_year}; TWh of primary energy"}])
    return out
