"""Loaders for the committed real-data extracts, with provenance."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from worldos.data.connectors.common import REAL_DATA, verify_manifest  # noqa: F401 (re-exported)
from worldos.population.empirical import (
    N_SINGLE, DemographicRates, EmpiricalFertility, EmpiricalMortality, YearRates, residual_migration,
    rogers_castro, shift_to_life_expectancy,
)

WPP = REAL_DATA / "wpp2024"


def available_countries() -> list[str]:
    return sorted(p.stem for p in (REAL_DATA / "naturalearth").glob("*.geojson"))


def check_integrity() -> dict[str, list[str]]:
    """Checksum every committed extract against its manifest. Empty lists mean intact."""
    return {d.name: verify_manifest(d) for d in REAL_DATA.iterdir() if (d / "manifest.json").exists()}


@lru_cache(maxsize=None)
def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def wpp_population(iso3: str, year: int) -> tuple[np.ndarray, np.ndarray]:
    """1 January population by single age 0..100 (persons), female and male."""
    src = "population_single_age_1950_2023.csv" if year <= 2023 else "population_single_age_2024_2100_medium.csv"
    d = _csv(str(WPP / src))
    d = d[(d.ISO3_code == iso3) & (d.Time == year)].sort_values("AgeGrpStart")
    if len(d) != N_SINGLE:
        raise KeyError(f"no WPP population for {iso3} {year}")
    return d.PopFemale.to_numpy() * 1000.0, d.PopMale.to_numpy() * 1000.0


def wpp_indicators(iso3: str) -> pd.DataFrame:
    d = _csv(str(WPP / "demographic_indicators.csv"))
    return d[d.ISO3_code == iso3].set_index("Time")


def _life_table(iso3: str, sex: str) -> pd.DataFrame:
    d = _csv(str(WPP / f"life_table_{sex}.csv"))
    return d[d.ISO3_code == iso3]


def wpp_rates(iso3: str, first_year: int = 1990, last_year: int = 2100) -> DemographicRates:
    lf, lm = _life_table(iso3, "female"), _life_table(iso3, "male")
    asfr = _csv(str(WPP / "asfr_single_age.csv"))
    asfr = asfr[asfr.ISO3_code == iso3]
    ind = wpp_indicators(iso3)
    rates = DemographicRates(iso3)
    last_lt = int(lf.Time.max())
    base_mort = None
    for year in range(first_year, last_year + 1):
        a = asfr[asfr.Time == year]
        if a.empty or year not in ind.index:
            continue
        fert_vec = np.zeros(N_SINGLE)
        fert_vec[a.AgeGrpStart.to_numpy()] = a.ASFR.to_numpy() / 1000.0
        # The single-age file covers ages 15-49 only; the published TFR also counts births to mothers <15 and 50+.
        # Rescale proportionally so that total fertility (hence births) matches the published TFR.
        if fert_vec.sum() > 0 and not np.isnan(ind.loc[year, "TFR"]):
            fert_vec *= float(ind.loc[year, "TFR"]) / fert_vec.sum()
        srb = float(ind.loc[year, "SRB"]) / 100.0
        fert = EmpiricalFertility(fert_vec, srb)
        if year <= last_lt:
            f_y, m_y = lf[lf.Time == year].sort_values("AgeGrpStart"), lm[lm.Time == year].sort_values("AgeGrpStart")
            mort = EmpiricalMortality(f_y.mx.to_numpy(), m_y.mx.to_numpy())
            base_mort = mort
            source = "estimate"
        else:
            mort = shift_to_life_expectancy(base_mort, float(ind.loc[year, "LExFemale"]), float(ind.loc[year, "LExMale"]))
            source = "projection"
        mig_f, mig_m = _migration(iso3, year, fert, lf, lm, ind)
        rates.years[year] = YearRates(mort, fert, mig_f, mig_m, source)
    return rates


def _migration(iso3, year, fert, lf, lm, ind) -> tuple[np.ndarray, np.ndarray]:
    try:
        pf, pm = wpp_population(iso3, year)
        nf, nm = wpp_population(iso3, year + 1)
        lt_f = lf[lf.Time == year].sort_values("AgeGrpStart")
        lt_m = lm[lm.Time == year].sort_values("AgeGrpStart")
        if lt_f.empty:
            raise KeyError
        women_mid = 0.5 * (pf + nf)
        births = float((fert.asfr_by_age * women_mid).sum())
        bf = births * fert.female_birth_share
        mf = residual_migration(pf, nf, lt_f.Lx.to_numpy() if "Lx" in lt_f else _Lx(lt_f), 100000.0, bf)
        mm = residual_migration(pm, nm, lt_m.Lx.to_numpy() if "Lx" in lt_m else _Lx(lt_m), 100000.0, births - bf)
        return mf, mm
    except KeyError:
        total = float(ind.loc[year, "NetMigrations"]) * 1000.0
        shape = rogers_castro(np.arange(N_SINGLE))
        shape /= shape.sum()
        return 0.5 * total * shape, 0.5 * total * shape


def _Lx(lt: pd.DataFrame) -> np.ndarray:
    """Person-years lived in each age interval from lx and mx: Lx = dx / mx (open interval: lx / mx)."""
    lx, mx = lt.lx.to_numpy(), lt.mx.to_numpy()
    dx = np.append(lx[:-1] - lx[1:], lx[-1])
    return np.where(mx > 0, dx / np.maximum(mx, 1e-12), lx)


def wdi(iso3: str) -> pd.DataFrame:
    path = REAL_DATA / "wdi" / "wdi.csv"
    if not path.exists():
        return pd.DataFrame(columns=["year", "variable", "value"])
    d = _csv(str(path))
    return d[d.iso3 == iso3]


def wdi_value(iso3: str, variable: str, year: int, lookback: int = 5) -> float | None:
    """Most recent non-missing value at or before ``year`` (within ``lookback`` years)."""
    d = wdi(iso3)
    d = d[(d.variable == variable) & (d.year <= year) & (d.year > year - lookback)].sort_values("year")
    return float(d.value.iloc[-1]) if len(d) else None


def wdi_series(iso3: str, variable: str) -> pd.Series:
    d = wdi(iso3)
    return d[d.variable == variable].set_index("year").value.sort_index()


def naturalearth(iso3: str) -> dict:
    return json.loads((REAL_DATA / "naturalearth" / f"{iso3}.geojson").read_text())


def manifest(source: str) -> dict:
    return json.loads((REAL_DATA / source / "manifest.json").read_text())


def data_path(*parts: str) -> Path:
    return REAL_DATA.joinpath(*parts)
