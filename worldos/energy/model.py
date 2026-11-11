"""Kaya-identity energy model with logistic diffusion of wind and solar.

    CO2 = sum_f  E * s_f * EF_f * k,      E = e_t * Y

* Y: real output; e_t: primary energy intensity, declining at a trend rate (INFERRED from history).
* s_f: shares of primary energy by source. Wind + solar ("VRE") follow a logistic diffusion curve
  s(t) = K / (1 + exp(-r (t - t_m)))  (Fisher & Pry 1971; technology-substitution literature),
  with the rate r fitted to the country's history, the ceiling K assumed, and t_m set so that the curve
  passes through the observed base-year share (anchoring; a faster rate then means faster diffusion
  from today's level). Nuclear, hydro, bioenergy
  and other renewables keep their base-year shares (assumption); fossil fuels fill the remainder in their
  base-year proportions, unless a coal phase-out year is set (coal falls linearly to zero, replaced by gas
  and oil in proportion).
* EF_f: IPCC 2006 Guidelines default CO2 emission factors (Vol. 2, Table 1.4), per unit of fuel energy:
  coal (bituminous) 94.6, crude oil 73.3, natural gas 56.1 tCO2/TJ; bioenergy counted as zero (biogenic).
* k: a single reconciliation factor so that base-year emissions match the Global Carbon Budget
  (absorbs non-energy CO2 such as cement, fuel-type mix within each category, and statistical
  differences); reported as a diagnostic.

Primary energy for non-fossil sources in the OWID data uses the substitution method (input-equivalent);
this affects shares, not fossil emissions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldos.data.connectors.common import REAL_DATA

SOURCES = ["coal", "oil", "gas", "nuclear", "hydro", "wind", "solar", "biofuel", "other_renewable"]
FOSSIL = ["coal", "oil", "gas"]
FIXED = ["nuclear", "hydro", "biofuel", "other_renewable"]
TJ_PER_MWH = 0.0036
EF_T_PER_MWH = {"coal": 94.6 * TJ_PER_MWH, "oil": 73.3 * TJ_PER_MWH, "gas": 56.1 * TJ_PER_MWH}


def load_energy(iso3: str) -> pd.DataFrame:
    d = pd.read_csv(REAL_DATA / "energy" / "energy_owid.csv")
    d = d[d.iso_code == iso3].set_index("year").sort_index()
    out = pd.DataFrame({s: d[f"{s}_consumption"] for s in SOURCES}).fillna(0.0)
    out["total"] = out[SOURCES].sum(axis=1)
    return out[out.total > 0]


def shares(df: pd.DataFrame) -> pd.DataFrame:
    return df[SOURCES].div(df.total, axis=0)


def fit_logistic(years: np.ndarray, share: np.ndarray, ceiling: float) -> tuple[float, float]:
    """Least squares on the logit of share/ceiling. Returns (rate r per year, midpoint year t_m)."""
    y = np.clip(share / ceiling, 1e-4, 1 - 1e-4)
    z = np.log(y / (1 - y))
    r, c = np.polyfit(years, z, 1)
    return float(r), float(-c / r)


def logistic(year: float | np.ndarray, ceiling: float, r: float, t_m: float):
    return ceiling / (1 + np.exp(-r * (np.asarray(year) - t_m)))


@dataclass
class EnergyFit:
    iso3: str
    base_year: int
    base_shares: dict
    intensity_decline: float  # annual, of primary energy per unit of real GDP
    vre_ceiling: float
    vre_rate: float
    vre_midpoint: float


def fit_energy(iso3: str, start: int, end: int, gdp: pd.Series, vre_ceiling: float = 0.6) -> EnergyFit | None:
    """Fit trends over [start, end]; None if the country lacks a source breakdown for those years."""
    e = load_energy(iso3)
    if len(e.loc[start:end]) < 5 or end not in e.index:
        return None
    sh = shares(e)
    yrs = np.arange(start, end + 1)
    intensity = (e.total / gdp).loc[start:end].dropna()
    decline = float(-np.polyfit(intensity.index, np.log(intensity.values), 1)[0])
    vre = (sh.wind + sh.solar).loc[start:end]
    r, tm = fit_logistic(yrs, vre.values, vre_ceiling)
    return EnergyFit(iso3, end, sh.loc[end].to_dict(), decline, vre_ceiling, r, tm)


def mix_for_year(fit: EnergyFit, year: float, coal_phaseout_year: int | None = None) -> dict:
    s = dict(fit.base_shares)
    vre_base = s["wind"] + s["solar"]
    # anchor the logistic at the observed base-year share: the midpoint follows from (share, rate), so
    # changing the rate means faster/slower diffusion from today's level, never a jump in the level
    if 0 < vre_base < fit.vre_ceiling and fit.vre_rate > 0:
        t_m = fit.base_year + np.log(fit.vre_ceiling / vre_base - 1) / fit.vre_rate
        vre = float(logistic(year, fit.vre_ceiling, fit.vre_rate, t_m))
    else:
        vre = vre_base
    # keep the wind/solar split of the base year
    ws = s["wind"] / vre_base if vre_base > 0 else 0.5
    out = {k: s[k] for k in FIXED}
    out["wind"], out["solar"] = vre * ws, vre * (1 - ws)
    fossil_room = max(0.0, 1.0 - sum(out.values()))
    fossil_base = sum(s[f] for f in FOSSIL)
    fos = {f: (s[f] / fossil_base if fossil_base > 0 else 1 / 3) for f in FOSSIL}
    if coal_phaseout_year is not None and year > fit.base_year:
        frac = float(np.clip((coal_phaseout_year - year) / max(coal_phaseout_year - fit.base_year, 1), 0, 1))
        coal = fos["coal"] * frac
        others = fos["oil"] + fos["gas"]
        scale = (1 - coal) / others if others > 0 else 0.0
        fos = {"coal": coal, "oil": fos["oil"] * scale, "gas": fos["gas"] * scale}
    for f in FOSSIL:
        out[f] = fossil_room * fos[f]
    return out


def fossil_co2_t(primary_mwh: float, mix: dict) -> float:
    return sum(primary_mwh * mix[f] * EF_T_PER_MWH[f] for f in FOSSIL)


def backtest(iso3: str, fit_window=(2008, 2018), test_end: int = 2023, vre_ceiling: float = 0.6) -> dict:
    """Fit on ``fit_window``, project to ``test_end`` driven by observed real GDP, compare with observations."""
    from worldos.countries import sources

    gdp = sources.wdi_series(iso3, "gdp_constant_2015_usd")
    a, b = fit_window
    fit = fit_energy(iso3, a, b, gdp, vre_ceiling)
    if fit is None:
        return None
    e = load_energy(iso3)
    sh = shares(e)
    co2 = pd.read_csv(REAL_DATA / "climate" / "emissions_owid_gcb.csv")
    co2 = co2[co2.iso_code == iso3].set_index("year")["co2"]
    base_mix = mix_for_year(fit, b)
    k = co2.loc[b] / (fossil_co2_t(e.total.loc[b] * 1e6, base_mix) / 1e6)
    rows = []
    for year in range(b + 1, test_end + 1):
        if year not in gdp.index or year not in e.index:
            continue
        energy = e.total.loc[b] * gdp.loc[year] / gdp.loc[b] * np.exp(-fit.intensity_decline * (year - b))
        mix = mix_for_year(fit, year)
        rows.append({"year": year, "energy_obs": e.total.loc[year], "energy_sim": energy,
                     "vre_obs": sh.wind.loc[year] + sh.solar.loc[year], "vre_sim": mix["wind"] + mix["solar"],
                     "co2_obs": co2.get(year, np.nan), "co2_sim": k * fossil_co2_t(energy * 1e6, mix) / 1e6})
    df = pd.DataFrame(rows).set_index("year")
    mape = lambda o, s: float((np.abs(df[s] / df[o] - 1)).mean())
    return {"iso3": iso3, "fit": fit, "reconciliation_k": float(k), "table": df,
            "energy_mape": mape("energy_obs", "energy_sim"), "co2_mape": mape("co2_obs", "co2_sim"),
            "vre_abs_error_end": float(abs(df.vre_sim.iloc[-1] - df.vre_obs.iloc[-1]))}


class EnergyEngine:
    """Monthly energy demand, source mix and fuel-based CO2 for a country world (config section ``energy``)."""

    name = "energy"
    order = 56  # after production; replaces the intensity-based EmissionsEngine when enabled
    period_months = 1
    phase = 0

    def step(self, world) -> None:
        c = world.config.energy
        if not c.enabled:
            return
        if world.energy_intensity is None:
            world.energy_intensity = c.intensity_mwh_per_unit
        else:
            world.energy_intensity *= float(np.exp(-c.intensity_decline / 12))
        fit = EnergyFit("", c.base_year, dict(zip(SOURCES, c.base_shares)), c.intensity_decline, c.vre_ceiling,
                        c.vre_rate, c.vre_midpoint)
        now = world.clock.now
        mix = mix_for_year(fit, now.year + (now.month - 0.5) / 12, c.coal_phaseout_year)
        mwh = world.energy_intensity * world.flows.units_produced
        world.flows.primary_energy_mwh = mwh
        world.flows.co2_emissions_t = c.reconciliation_k * fossil_co2_t(mwh, mix)
        world.energy_mix = mix


def energy_report(countries=("IND", "USA", "DEU", "JPN", "BRA", "NGA")) -> str:
    L = ["# Energy model: out-of-sample backtest", "",
         "> Kaya identity with logistic wind+solar diffusion; IPCC 2006 default emission factors. Fitted on 2008-2018,",
         "> projected 2019-2023 driven by observed real GDP (WDI), compared with OWID/Energy Institute energy and",
         "> Global Carbon Budget fossil CO2. Generated by `python -m worldos energy validate`.", "",
         "| country | reconciliation k | energy-intensity decline /yr | energy MAPE | CO2 MAPE | wind+solar 2023 model / observed |",
         "|---|---|---|---|---|---|"]
    for iso in countries:
        r = backtest(iso)
        if r is None:
            L.append(f"| {iso} | n/a | n/a | n/a | n/a | no source breakdown in the dataset; intensity-based CO2 used |")
            continue
        t = r["table"].iloc[-1]
        L.append(f"| {iso} | {r['reconciliation_k']:.3f} | {r['fit'].intensity_decline:.4f} | {r['energy_mape']:.1%} | "
                 f"{r['co2_mape']:.1%} | {t.vre_sim:.3f} / {t.vre_obs:.3f} |")
    L += ["", "## Reading the results", "",
          "- k close to 1 means IPCC fuel factors applied to primary fossil energy explain fossil CO2; deviations absorb",
          "  cement and other non-energy CO2 and fuel-quality differences.",
          "- Energy demand and CO2 are projected within a few percent out of sample for most countries; 2020 (COVID) is",
          "  inside the test window and is captured only through observed GDP.",
          "- Logistic diffusion fitted on an early, steep phase extrapolates too fast for some countries (Japan, Brazil):",
          "  diffusion parameters are weakly identified before the inflection point. Treat wind+solar projections as",
          "  scenario assumptions, not forecasts.",
          "- Nuclear, hydro and bioenergy shares are held at base-year values (e.g. Germany's nuclear phase-out is not",
          "  represented in the backtest).", ""]
    return "\n".join(L)
