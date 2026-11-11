"""Build a WORLD-OS world for a real country from the committed real-data extracts.

What is real (OBSERVED/ESTIMATED by the source): population by sex and single age, mortality,
fertility and net-migration rates by year (UN WPP 2024), country/region boundaries and cities
(Natural Earth), macro targets (World Bank WDI).
What is INFERRED here: spatial allocation of households, labour-force participation within the
model's working-age band, productivity level and trend, Taylor-rule natural unemployment.
What is ASSUMED: behavioural parameters (config defaults), pension age and inflation target
(institutions.toml), education distribution.
What is CALIBRATED: the income tax rate, chosen so the simulated unemployment rate over the first
years matches the observed (ILO-modelled) rate. It absorbs every omitted fiscal item, so it is
an effective model parameter, not an estimate of the statutory tax rate.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

from worldos.config import WorldConfig, apply_overrides
from worldos.core.actions import ActionKernel
from worldos.core.clock import EventScheduler, SimDate, SimulationClock
from worldos.core.provenance import ValueKind
from worldos.core.rng import RandomStreams
from worldos.core.world import World
from worldos.countries import sources
from worldos.countries.geography import build_country_geography
from worldos.economy.initialize import initialize_economy
from worldos.economy.state import BankState, CentralBankState, FirmTable, GovernmentState
from worldos.institutions.actions import register_default_actions
from worldos.institutions.policies import FiscalRulePolicy, TaylorRulePolicy
from worldos.population.synthesis import synthesize_population
from worldos.build import neutral_debt_target  # noqa: F401 (re-exported)


@dataclass
class ParameterRecord:
    path: str
    value: float
    kind: ValueKind
    source: str


@dataclass
class CountryInfo:
    iso3: str
    name: str
    base_year: int
    geography: dict
    parameters: list[ParameterRecord] = field(default_factory=list)
    calibration: dict = field(default_factory=dict)


def unit_price_factor(cfg: WorldConfig) -> float:
    """Initial price per unit of output relative to unit labour cost (full-cost pricing fixed point)."""
    ec = cfg.economy
    mu, dky = ec.initial_markup, ec.depreciation_rate * ec.capital_output_ratio
    return (1 + mu) / (1 - (1 + mu) * dky)


def _owid_co2(iso3: str):
    import pandas as pd

    d = pd.read_csv(sources.data_path("climate", "emissions_owid_gcb.csv"))
    return d[d.iso_code == iso3].set_index("year")["co2"].dropna()


def institutions(iso3: str) -> dict:
    text = resources.files("worldos.countries").joinpath("institutions.toml").read_text()
    return tomllib.loads(text).get(iso3, {})


def _monthly_grid(single_age: np.ndarray, n_ages: int) -> np.ndarray:
    """Spread counts by single year of age (0..100+) uniformly over month-ages (100+ over 100-104)."""
    out = np.zeros(n_ages)
    for age, v in enumerate(single_age):
        span = 12 if age < 100 else 60
        out[age * 12: age * 12 + span] += v / span
    return out


def country_config(iso3: str, base_year: int, n_records: int, n_firms: int,
                   overrides: dict | None = None) -> tuple[WorldConfig, list[ParameterRecord]]:
    recs: list[ParameterRecord] = []
    ov: dict = {}

    def put(path: str, value: float, kind: ValueKind, source: str) -> None:
        ov[path] = value
        recs.append(ParameterRecord(path, float(value), kind, source))

    pf, pm = sources.wpp_population(iso3, base_year)
    total = pf.sum() + pm.sum()
    put("population.represented_population", total, ValueKind.ESTIMATED, f"UN WPP 2024, 1 Jan {base_year}")
    ov["population.n_person_records"] = n_records
    ov["economy.n_firms"] = n_firms
    ov["simulation.start"] = f"{base_year}-01"
    inst = institutions(iso3)
    pension = int(inst.get("pension_age", 65))
    put("population.pension_age", pension, ValueKind.ASSUMED, "countries/institutions.toml")
    put("central_bank.inflation_target", inst.get("inflation_target", 0.02), ValueKind.ASSUMED, "countries/institutions.toml")

    wdi = lambda v: sources.wdi_value(iso3, v, base_year)
    lfpr = wdi("labour_force_participation_15plus")
    if lfpr is not None:
        p15 = (pf + pm)[15:].sum()
        band = (pf + pm)[17:pension].sum()  # model labour force: after school leaving, before pension
        part = float(np.clip(lfpr / 100 * p15 / band, 0.3, 0.98))
        put("population.participation_rate", part, ValueKind.INFERRED, "WDI SL.TLF.CACT.ZS rescaled to the model's working-age band")
    u = wdi("unemployment_rate_ilo")
    if u is not None:
        put("economy.initial_unemployment", u / 100, ValueKind.ESTIMATED, "WDI SL.UEM.TOTL.ZS (ILO modelled)")
        hist = sources.wdi_series(iso3, "unemployment_rate_ilo")
        hist = hist[(hist.index <= base_year) & (hist.index > base_year - 10)]
        put("central_bank.natural_unemployment", float(hist.mean()) / 100, ValueKind.INFERRED, "10-year mean of WDI unemployment")
    g = wdi("government_consumption_share_gdp")
    if g is not None:
        put("government.spending_share", g / 100, ValueKind.ESTIMATED, "WDI NE.CON.GOVT.ZS")
    gdp = wdi("gdp_constant_2015_usd")
    epop = wdi("employment_to_population_15plus")
    if gdp is not None and epop is not None:
        employed = epop / 100 * (pf + pm)[15:].sum()
        base_cfg = apply_overrides(WorldConfig(), overrides or {})
        mean_h = float(np.dot(base_cfg.population.education_distribution, base_cfg.economy.education_productivity))
        w0 = gdp / employed / 12 / unit_price_factor(base_cfg) / mean_h
        put("economy.initial_wage", w0, ValueKind.INFERRED, "WDI GDP (constant 2015 USD) per employed person, monthly")
        put("economy.initial_productivity", w0, ValueKind.INFERRED, "set equal to wage: one good costs ~1 USD of labour")
    gpe = sources.wdi_series(iso3, "gdp_per_person_employed_ppp_2021")
    gpe = gpe[(gpe.index <= base_year) & (gpe.index >= base_year - 10)]
    if len(gpe) >= 5:
        years = gpe.index[-1] - gpe.index[0]
        growth = float(np.clip((gpe.iloc[-1] / gpe.iloc[0]) ** (1 / years) - 1, -0.02, 0.08))
        put("economy.productivity_growth", growth, ValueKind.INFERRED, "CAGR of WDI GDP per person employed, last 10 years")
    # emissions: intensity per unit of real output
    co2 = _owid_co2(iso3)
    if gdp is not None and base_year in co2.index:
        unit_price = unit_price_factor(apply_overrides(WorldConfig(), overrides or {}))  # USD per unit of output
        intensity = co2.loc[base_year] * 1e6 / gdp * unit_price
        put("climate.emissions_intensity", intensity, ValueKind.INFERRED,
            "GCB fossil CO2 (via OWID) / WDI GDP constant 2015 USD, base year, per unit of real output")
        gdp_s = sources.wdi_series(iso3, "gdp_constant_2015_usd")
        y0 = base_year - 10
        if y0 in co2.index and y0 in gdp_s.index and base_year in gdp_s.index:
            i0 = co2.loc[y0] / gdp_s.loc[y0]
            i1 = co2.loc[base_year] / gdp_s.loc[base_year]
            put("climate.decarbonisation_rate", float(-np.log(i1 / i0) / 10), ValueKind.INFERRED,
                f"trend of CO2 intensity of GDP {y0}-{base_year}")
    target = ov.get("central_bank.inflation_target", 0.02)
    put("central_bank.initial_policy_rate", 0.01 + target, ValueKind.ASSUMED, "neutral real rate 1% + inflation target")
    cfg = apply_overrides(WorldConfig(), ov)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg, recs


def build_country_world(iso3: str, base_year: int = 2023, n_records: int = 40_000, n_firms: int = 400, seed: int = 1,
                        overrides: dict | None = None, rates=None, calibrated: bool = True) -> World:
    """Build a real-country world. With ``calibrated`` the stored calibration (data/calibration/<ISO>.json,
    produced by ``worldos country calibrate``) sets the effective income-tax rate."""
    overrides = dict(overrides or {})
    calib = None
    if calibrated and "government.income_tax_rate" not in overrides:
        from worldos.countries.calibrate import load_calibration

        calib = load_calibration(iso3)
        if calib is not None:
            overrides["government.income_tax_rate"] = calib["income_tax_rate"]
            if calib.get("capital_output_ratio") and "economy.capital_output_ratio" not in overrides:
                overrides["economy.capital_output_ratio"] = calib["capital_output_ratio"]
    cfg, recs = country_config(iso3, base_year, n_records, n_firms, overrides)
    if calib is not None:
        recs.append(ParameterRecord("government.income_tax_rate", calib["income_tax_rate"], ValueKind.ASSUMED,
                                    f"CALIBRATED to WDI unemployment {calib['target_unemployment']:.4f} "
                                    f"(achieved {calib['achieved_unemployment']:.4f})"))
        if calib.get("capital_output_ratio"):
            recs.append(ParameterRecord("economy.capital_output_ratio", calib["capital_output_ratio"], ValueKind.ASSUMED,
                                        f"CALIBRATED to WDI investment share {calib['target_investment_share']:.4f} "
                                        f"(achieved {calib['achieved_investment_share']:.4f})"))
    rng = RandomStreams(seed)
    urban = sources.wdi_value(iso3, "urban_population_share", base_year)
    urban_share = (urban / 100) if urban is not None else 0.5
    geo, layout, geo_info = build_country_geography(iso3, sources.naturalearth(iso3), urban_share,
                                                    grid=sources.kontur_grid(iso3))
    rates = rates or sources.wpp_rates(iso3)
    yr = rates.for_year(base_year)
    pf, pm = sources.wpp_population(iso3, base_year)
    n_ages = yr.mortality.n_ages
    persons, households = synthesize_population(cfg.population, _monthly_grid(pf, n_ages), _monthly_grid(pm, n_ages),
                                                layout, rng.get("init.population"))
    kernel = ActionKernel()
    register_default_actions(kernel)
    world = World(config=cfg, clock=SimulationClock(SimDate.parse(cfg.simulation.start)), rng=rng, geo=geo,
                  layout=layout, mortality=yr.mortality, fertility=yr.fertility, persons=persons,
                  households=households, firms=FirmTable(), bank=BankState(), government=GovernmentState(),
                  central_bank=CentralBankState(), kernel=kernel, events=EventScheduler(),
                  run_id=f"{iso3}-{base_year}-seed{seed}")
    world.demographic_rates = rates
    initialize_economy(world, rng.get("init.economy"))
    world.strategic_agents = {"central_bank": TaylorRulePolicy(cfg.central_bank),
                              "government": FiscalRulePolicy(cfg.government, inflation_target=cfg.central_bank.inflation_target)}
    name = geo[iso3.lower()].name
    world.country = CountryInfo(iso3, name, base_year, geo_info, recs, calib or {})
    neutral_debt_target(world)
    return world
