"""UN World Population Prospects 2024 connector (bulk CSV files, CC BY 3.0 IGO).

The full files are large (60-300 MB compressed), so they are streamed and filtered on the fly to
the requested countries; only the filtered extracts are stored.

Citation: United Nations, Department of Economic and Social Affairs, Population Division (2024).
World Population Prospects 2024. Licence: CC BY 3.0 IGO.
"""

from __future__ import annotations

import csv
import gzip
import io
import urllib.request
from pathlib import Path

from worldos.data.connectors.common import DEFAULT_COUNTRIES, REAL_DATA, write_manifest

BASE = "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/"
LICENSE = "CC BY 3.0 IGO"

# file -> (output name, columns to keep, extra row filter)
FILES = {
    "WPP2024_Population1JanuaryBySingleAgeSex_Medium_1950-2023.csv.gz":
        ("population_single_age_1950_2023.csv", ["ISO3_code", "Time", "AgeGrpStart", "PopMale", "PopFemale"], None),
    "WPP2024_Population1JanuaryBySingleAgeSex_Medium_2024-2100.csv.gz":
        ("population_single_age_2024_2100_medium.csv", ["ISO3_code", "Time", "AgeGrpStart", "PopMale", "PopFemale"], None),
    "WPP2024_Fertility_by_Age1.csv.gz":
        ("asfr_single_age.csv", ["ISO3_code", "Time", "AgeGrpStart", "ASFR"], lambda r: r["Variant"] == "Medium"),
    "WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv.gz":
        ("life_table_female.csv", ["ISO3_code", "Time", "AgeGrpStart", "mx", "qx", "lx", "ex"], None),
    "WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv.gz":
        ("life_table_male.csv", ["ISO3_code", "Time", "AgeGrpStart", "mx", "qx", "lx", "ex"], None),
    "WPP2024_Demographic_Indicators_Medium.csv.gz":
        ("demographic_indicators.csv", ["ISO3_code", "Location", "Time", "TPopulation1Jan", "TFR", "MAC", "SRB",
                                        "LEx", "LExMale", "LExFemale", "CBR", "CDR", "NetMigrations", "CNMR",
                                        "MedianAgePop", "IMR"], None),
}


def fetch_wpp(countries: list[str] | None = None, min_year: int = 1990, out_dir: Path | None = None) -> Path:
    countries = set(countries or DEFAULT_COUNTRIES)
    out = out_dir or REAL_DATA / "wpp2024"
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for fname, (outname, cols, extra) in FILES.items():
        url = BASE + fname
        n = 0
        with urllib.request.urlopen(url, timeout=600) as resp, open(out / outname, "w", newline="") as fh:
            reader = csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=resp), encoding="utf-8-sig"))
            writer = csv.writer(fh)
            writer.writerow(cols)
            for row in reader:
                if row["ISO3_code"] not in countries or int(row["Time"]) < min_year:
                    continue
                if extra is not None and not extra(row):
                    continue
                writer.writerow([row[c] for c in cols])
                n += 1
        entries.append({"file": outname, "url": url, "rows": n,
                        "filter": f"ISO3 in {sorted(countries)}, year >= {min_year}"})
    write_manifest(out, "UN DESA Population Division, World Population Prospects 2024", LICENSE, entries,
                   notes="Population in thousands; ASFR per 1000 women; mx/qx per person-year; ages in years.")
    return out
