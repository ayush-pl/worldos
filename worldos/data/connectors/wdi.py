"""World Bank World Development Indicators connector (public API v2, CC BY 4.0).

Citation: World Bank, World Development Indicators. Licence: CC BY 4.0.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

from worldos.data.connectors.common import DEFAULT_COUNTRIES, REAL_DATA, write_manifest

API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json&per_page=20000&date={start}:{end}"
LICENSE = "CC BY 4.0"

# indicator code -> (short name, unit)
INDICATORS = {
    "SP.POP.TOTL": ("population", "persons"),
    "NY.GDP.MKTP.CD": ("gdp_current_usd", "USD"),
    "NY.GDP.MKTP.KD": ("gdp_constant_2015_usd", "USD"),
    "NY.GDP.MKTP.KD.ZG": ("gdp_real_growth", "%"),
    "NY.GDP.PCAP.KD": ("gdp_per_capita_constant_2015_usd", "USD/person"),
    "FP.CPI.TOTL.ZG": ("cpi_inflation", "%"),
    "SL.UEM.TOTL.ZS": ("unemployment_rate_ilo", "%"),
    "SL.TLF.CACT.ZS": ("labour_force_participation_15plus", "%"),
    "SL.EMP.TOTL.SP.ZS": ("employment_to_population_15plus", "%"),
    "NE.CON.GOVT.ZS": ("government_consumption_share_gdp", "%"),
    "NE.CON.PRVT.ZS": ("household_consumption_share_gdp", "%"),
    "NE.GDI.TOTL.ZS": ("gross_capital_formation_share_gdp", "%"),
    "NE.EXP.GNFS.ZS": ("exports_share_gdp", "%"),
    "NE.IMP.GNFS.ZS": ("imports_share_gdp", "%"),
    "GC.TAX.TOTL.GD.ZS": ("tax_revenue_share_gdp", "%"),
    "GC.REV.XGRT.GD.ZS": ("government_revenue_share_gdp", "%"),
    "GC.XPN.TOTL.GD.ZS": ("government_expense_share_gdp", "%"),
    "GC.DOD.TOTL.GD.ZS": ("central_government_debt_share_gdp", "%"),
    "SI.POV.GINI": ("gini_index", "index"),
    "SP.URB.TOTL.IN.ZS": ("urban_population_share", "%"),
    "SL.GDP.PCAP.EM.KD": ("gdp_per_person_employed_ppp_2021", "USD"),
    "FR.INR.LEND": ("lending_interest_rate", "%"),
    "SP.DYN.TFRT.IN": ("total_fertility_rate", "births/woman"),
    "SP.DYN.LE00.IN": ("life_expectancy_at_birth", "years"),
}


def _get(url: str, retries: int = 4) -> list:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception:  # network hiccup: exponential backoff
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))
    return []


def fetch_wdi(countries: list[str] | None = None, start: int = 1990, end: int = 2024, out_dir: Path | None = None) -> Path:
    countries = countries or DEFAULT_COUNTRIES
    out = out_dir or REAL_DATA / "wdi"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for code, (name, unit) in INDICATORS.items():
        payload = _get(API.format(countries=";".join(countries), indicator=code, start=start, end=end))
        if len(payload) < 2 or payload[1] is None:
            continue
        for rec in payload[1]:
            if rec["value"] is None:
                continue
            rows.append([rec["countryiso3code"], int(rec["date"]), code, name, unit, rec["value"]])
    rows.sort()
    with open(out / "wdi.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "year", "indicator", "variable", "unit", "value"])
        w.writerows(rows)
    write_manifest(out, "World Bank, World Development Indicators (API v2)", LICENSE,
                   [{"file": "wdi.csv", "url": API.split("?")[0], "rows": len(rows),
                     "filter": f"countries {countries}, years {start}-{end}"}])
    return out
