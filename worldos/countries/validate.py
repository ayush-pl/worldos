"""Country validation report: what the real-country world reproduces, and what it does not.

1. Demographic hindcast: start from WPP 1 Jan 2000, run the microsimulation with WPP's year-specific
   rates (mortality, fertility, residual migration), compare with WPP estimates up to 1 Jan 2024.
   Because the rates come from the same source, this tests the *engine* (event timing, migration
   handling, weighting) rather than demographic forecasting skill.
2. Projection benchmark: from 1 Jan 2024 to 2050 with WPP-projected TFR/ASFR, life expectancy
   (proportional mortality shift) and net migration (Rogers-Castro age pattern), compared with the
   WPP 2024 medium variant. Agreement shows consistency with WPP, not independent predictive skill.
3. Economy: calibrated run; targeted moment (unemployment) and untargeted moments vs WDI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldos.core.simulation import Simulation
from worldos.countries import sources
from worldos.countries.build import build_country_world
from worldos.population.demography import CohortProjection, DemographyEngine, aggregate_age_sex

GROUPS = [(0, 15), (15, 65), (65, 111)]


def _groups(f: np.ndarray, m: np.ndarray, monthly: bool) -> np.ndarray:
    tot = f + m
    ages = np.arange(len(tot)) / (12 if monthly else 1)
    return np.array([tot[(ages >= a) & (ages < b)].sum() for a, b in GROUPS])


def demographic_run(iso3: str, start_year: int, end_year: int, n_records: int = 60_000, seed: int = 1,
                    checkpoints: list[int] | None = None) -> pd.DataFrame:
    rates = sources.wpp_rates(iso3)
    w = build_country_world(iso3, start_year, n_records, 20, seed, rates=rates, calibrated=False)
    cohort = CohortProjection.from_persons(w.persons, w.mortality, w.fertility)
    engine = DemographyEngine()
    checkpoints = checkpoints or list(range(start_year + 5, end_year + 1, 5)) + [end_year]
    rows = []
    for year in range(start_year, end_year):
        yr = rates.for_year(year)
        cohort.set_rates(yr.mortality, yr.fertility)
        for _ in range(12):
            w.begin_tick()
            engine.step(w)
            w.clock.advance()
            cohort.step(yr.migration_female, yr.migration_male)
        if year + 1 in checkpoints:
            f, m = aggregate_age_sex(w.persons, w.mortality.n_ages)
            wf, wm = sources.wpp_population(iso3, year + 1)
            micro, macro, obs = _groups(f, m, True), _groups(cohort.female, cohort.male, True), _groups(wf, wm, False)
            rows.append({
                "year": year + 1, "wpp_total": obs.sum(), "micro_total": micro.sum(), "cohort_total": macro.sum(),
                "micro_error": micro.sum() / obs.sum() - 1, "cohort_error": macro.sum() / obs.sum() - 1,
                **{f"share_{a}_{b}_wpp": v / obs.sum() for (a, b), v in zip(GROUPS, obs)},
                **{f"share_{a}_{b}_micro": v / micro.sum() for (a, b), v in zip(GROUPS, micro)},
                "age_tvd": 0.5 * np.abs(micro / micro.sum() - obs / obs.sum()).sum(),
                "source": rates.for_year(year).source,
            })
    return pd.DataFrame(rows)


def economy_moments(iso3: str, seeds=(1, 2, 3), months: int = 60, n_records: int = 20_000, n_firms: int = 200) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        w = build_country_world(iso3, 2023, n_records, n_firms, seed)
        df = Simulation(w).run(months).metrics_frame().iloc[12:]
        frames.append({
            "unemployment_rate": df.unemployment_rate.mean(),
            "household_consumption_share_gdp": (df.consumption / df.gdp_nominal).mean(),
            "government_consumption_share_gdp": (df.gov_purchases / df.gdp_nominal).mean(),
            "cpi_inflation": df.inflation_yoy.dropna().mean(),
            "employment_to_population_15plus": np.nan,
            "real_gdp_growth": (df.gdp_real.iloc[-12:].sum() / df.gdp_real.iloc[:12].sum()) ** (1 / ((len(df) - 12) / 12)) - 1,
            "gini_income": df.gini_income.mean(),
            "gdp_per_capita_usd": (12 * df.gdp_nominal / df.population).iloc[:12].mean(),
        })
    model = pd.DataFrame(frames)
    obs = {
        "unemployment_rate": sources.wdi_value(iso3, "unemployment_rate_ilo", 2023),
        "household_consumption_share_gdp": sources.wdi_value(iso3, "household_consumption_share_gdp", 2023),
        "government_consumption_share_gdp": sources.wdi_value(iso3, "government_consumption_share_gdp", 2023),
        "cpi_inflation": sources.wdi_series(iso3, "cpi_inflation").loc[2014:2023].mean(),
        "real_gdp_growth": sources.wdi_series(iso3, "gdp_real_growth").loc[2014:2023].mean(),
        "gini_income": sources.wdi_value(iso3, "gini_index", 2023, lookback=10),
        "gdp_per_capita_usd": sources.wdi_value(iso3, "gdp_per_capita_constant_2015_usd", 2023),
    }
    scale = {"gini_income": 100, "gdp_per_capita_usd": 1}
    rows = []
    for k, v in obs.items():
        o = None if v is None or pd.isna(v) else v / scale.get(k, 100)
        rows.append({"moment": k, "targeted": k in ("unemployment_rate", "government_consumption_share_gdp",
                                                    "gdp_per_capita_usd"),
                     "observed": o, "model_mean": model[k].mean(), "model_sd": model[k].std(ddof=1)})
    return pd.DataFrame(rows)


def render_report(iso3: str, hindcast: pd.DataFrame, projection: pd.DataFrame, econ: pd.DataFrame) -> str:
    w = build_country_world(iso3, 2023, 2000, 20, 1)
    info = w.country
    L = [f"# Country validation report: {info.name} ({iso3})", "",
         "> Generated by `python -m worldos country validate`. Model experiments on a calibrated, simplified",
         "> economy; not forecasts. Demographic agreement with WPP is consistency with the source, not predictive skill.",
         "", "## Data and parameters", "",
         f"Geography: Natural Earth; H3 resolution {info.geography['h3_resolution']}, {info.geography['n_cells']} cells, "
         f"{info.geography['n_regions']} admin-1 regions, {info.geography['n_cities']} cities, urban share "
         f"{info.geography['urban_share']:.3f} (WDI).", "",
         "| parameter | value | provenance | source |", "|---|---|---|---|"]
    for r in info.parameters:
        L.append(f"| `{r.path}` | {r.value:.6g} | {r.kind.value} | {r.source} |")
    L += ["", "## 1. Demographic hindcast 2000 → 2024 (WPP year-specific rates)", "",
          "| 1 Jan | WPP total | micro total | micro error | cohort error | age TVD | 0-14 WPP / micro | 65+ WPP / micro |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in hindcast.iterrows():
        L.append(f"| {r.year} | {r.wpp_total / 1e6:,.2f} M | {r.micro_total / 1e6:,.2f} M | {r.micro_error:+.3%} | "
                 f"{r.cohort_error:+.3%} | {r.age_tvd:.4f} | {r.share_0_15_wpp:.3f} / {r.share_0_15_micro:.3f} | "
                 f"{r.share_65_111_wpp:.3f} / {r.share_65_111_micro:.3f} |")
    L += ["", "## 2. Projection benchmark 2024 → 2050 (vs WPP 2024 medium variant)", "",
          "| 1 Jan | WPP medium | micro | micro error | cohort error | age TVD | 65+ WPP / micro |", "|---|---|---|---|---|---|---|"]
    for _, r in projection.iterrows():
        L.append(f"| {r.year} | {r.wpp_total / 1e6:,.2f} M | {r.micro_total / 1e6:,.2f} M | {r.micro_error:+.3%} | "
                 f"{r.cohort_error:+.3%} | {r.age_tvd:.4f} | {r.share_65_111_wpp:.3f} / {r.share_65_111_micro:.3f} |")
    L += ["", "## 3. Economy: calibrated model vs World Bank WDI", "",
          "Model: mean of months 12-60 across 3 seeds (± sd across seeds). Observed: 2023 value, or the 2014-2023 mean "
          "for inflation and growth.", "", "| moment | targeted? | observed | model |", "|---|---|---|---|"]
    for _, r in econ.iterrows():
        o = "n/a" if r.observed is None or pd.isna(r.observed) else f"{r.observed:.4g}"
        L.append(f"| {r.moment} | {'yes' if r.targeted else 'no'} | {o} | {r.model_mean:.4g} ± {r.model_sd:.2g} |")
    L += ["", "### Reading the economic comparison", "",
          "- Alpha has no investment, housing or foreign sector, so household + government consumption must equal "
          "~100% of model GDP; the real household consumption share is lower. This gap is structural, not a calibration error.",
          "- The Gini comparison is indicative only: WDI Ginis mix income and consumption surveys, and the model "
          "lacks within-record heterogeneity and capital income dispersion.",
          "- Growth in the model is the productivity trend plus labour-force change; it is not fitted to GDP growth.", ""]
    return "\n".join(L)


def country_report(iso3: str, n_records: int = 60_000) -> str:
    hind = demographic_run(iso3, 2000, 2024, n_records, checkpoints=[2005, 2010, 2015, 2020, 2024])
    proj = demographic_run(iso3, 2024, 2050, n_records, checkpoints=[2030, 2040, 2050])
    econ = economy_moments(iso3)
    return render_report(iso3, hind, proj, econ)
