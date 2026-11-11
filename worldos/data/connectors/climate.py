"""Historical climate records for validating the climate module.

* Global Carbon Budget emissions via Our World in Data's CO2 dataset (CC BY 4.0).
* NOAA GML Mauna Loa annual mean CO2 (US Government, public domain).
* HadCRUT5 global annual temperature anomaly, analysis summary series (Open Government Licence v3).
* IPCC AR6 WG1 effective radiative forcing 1750-2019 (Smith et al. 2021, Chapter 7 data repository;
  code MIT, data published under CC BY 4.0).
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

from worldos.data.connectors.common import DEFAULT_COUNTRIES, REAL_DATA, write_manifest

OWID = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
NOAA = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"
HADCRUT = ("https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/"
           "HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv")


AR6_ERF = "https://raw.githubusercontent.com/IPCC-WG1/Chapter-7/main/data_output/AR6_ERF_1750-2019.csv"


def _text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read().decode("utf-8")


def fetch_climate(countries: list[str] | None = None, out_dir: Path | None = None) -> Path:
    countries = set(countries or DEFAULT_COUNTRIES)
    out = out_dir or REAL_DATA / "climate"
    out.mkdir(parents=True, exist_ok=True)
    keep = ["iso_code", "country", "year", "co2", "land_use_change_co2", "co2_including_luc", "gdp", "population"]
    rows = []
    for r in csv.DictReader(io.StringIO(_text(OWID))):
        if r["country"] == "World" or r["iso_code"] in countries:
            rows.append([r.get(k, "") for k in keep])
    with open(out / "emissions_owid_gcb.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keep)
        w.writerows(rows)
    noaa = [l for l in _text(NOAA).splitlines() if l and not l.startswith("#")]
    (out / "co2_mauna_loa_annual.csv").write_text("\n".join(noaa) + "\n")
    (out / "hadcrut5_global_annual.csv").write_text(_text(HADCRUT))
    (out / "ar6_erf_1750_2019.csv").write_text(_text(AR6_ERF))
    write_manifest(out, "Global Carbon Budget (via OWID), NOAA GML, Met Office HadCRUT5", "mixed: see files", [
        {"file": "emissions_owid_gcb.csv", "url": OWID, "rows": len(rows), "license": "CC BY 4.0",
         "filter": f"World + {sorted(countries)}; CO2 in MtCO2/yr"},
        {"file": "co2_mauna_loa_annual.csv", "url": NOAA, "license": "US Government work (public domain)"},
        {"file": "hadcrut5_global_annual.csv", "url": HADCRUT, "license": "Open Government Licence v3",
         "filter": "anomaly relative to 1961-1990"},
        {"file": "ar6_erf_1750_2019.csv", "url": AR6_ERF, "license": "CC BY 4.0 (data); MIT (repository code)",
         "filter": "effective radiative forcing, W m-2, by agent"},
    ])
    return out
