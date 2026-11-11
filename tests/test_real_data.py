"""Tests on the committed real-data extracts (no network access needed)."""

import numpy as np
import pytest

from worldos.climate.history import calibrate, historical_inputs
from worldos.climate.model import ClimateParams, _alpha, run_climate
from worldos.core.simulation import Simulation
from worldos.countries import sources
from worldos.countries.build import build_country_world
from worldos.countries.geography import build_country_geography
from worldos.countries.validate import demographic_run
from worldos.population.schedules import FEMALE, MALE

COUNTRIES = ["IND", "USA", "DEU", "JPN", "BRA", "NGA"]


def test_extract_checksums_match_manifests():
    assert all(not changed for changed in sources.check_integrity().values())


@pytest.mark.parametrize("iso", COUNTRIES)
def test_empirical_schedules_reproduce_wpp_indicators(iso):
    rates = sources.wpp_rates(iso)
    ind = sources.wpp_indicators(iso)
    for year in (2000, 2023):
        yr = rates.for_year(year)
        assert yr.fertility.tfr == pytest.approx(ind.loc[year, "TFR"], abs=0.01)
        assert yr.mortality.life_expectancy(FEMALE) == pytest.approx(ind.loc[year, "LExFemale"], abs=0.15)
        assert yr.mortality.life_expectancy(MALE) == pytest.approx(ind.loc[year, "LExMale"], abs=0.15)
    proj = rates.for_year(2050)
    assert proj.source == "projection"
    assert proj.mortality.life_expectancy(FEMALE) == pytest.approx(ind.loc[2050, "LExFemale"], abs=0.01)


def test_residual_migration_close_to_reported_net_migration():
    rates, ind = sources.wpp_rates("USA"), sources.wpp_indicators("USA")
    est = np.mean([rates.for_year(y).migration_female.sum() + rates.for_year(y).migration_male.sum() for y in range(2010, 2020)])
    rep = ind.loc[2010:2019, "NetMigrations"].mean() * 1000
    assert est == pytest.approx(rep, rel=0.15)


def test_cohort_hindcast_matches_wpp():
    df = demographic_run("DEU", 2000, 2010, n_records=20_000, checkpoints=[2010])
    assert abs(df.cohort_error.iloc[-1]) < 0.005
    assert abs(df.micro_error.iloc[-1]) < 0.02


@pytest.mark.parametrize("iso", ["DEU", "JPN"])
def test_real_geography(iso):
    geo, layout, info = build_country_geography(iso, sources.naturalearth(iso), 0.8)
    country = geo[iso.lower()]
    regions = geo.children(iso.lower())
    assert set().union(*(r.cells for r in regions)) == set(country.cells)
    assert sum(len(r.cells) for r in regions) == len(country.cells)
    assert layout.city_share.sum() == pytest.approx(1.0)
    capital = {"DEU": "Berlin", "JPN": "Tokyo"}[iso]
    assert any(geo[c].name == capital for c in layout.city_ids)


def test_country_world_runs_with_invariants_and_real_totals():
    w = build_country_world("BRA", 2023, n_records=5000, n_firms=60)
    pf, pm = sources.wpp_population("BRA", 2023)
    assert w.persons.weight[w.persons.alive].sum() == pytest.approx(pf.sum() + pm.sum(), rel=1e-9)
    kinds = {r.kind.value for r in w.country.parameters}
    assert {"estimated", "inferred", "assumed"} <= kinds
    sim = Simulation(w).run(24)
    assert sim.violations == []
    df = sim.metrics_frame()
    assert df.co2_mt_annualised.iloc[-1] > 0
    assert (df.emigrants.sum() + df.immigrants.sum()) > 0


def test_fair_alpha_solver_inverts_iirf100():
    from worldos.climate.model import FAIR_TAU, JOOS_A

    for target in (25.0, 35.0, 60.0):
        a = _alpha(target)
        assert (JOOS_A * a * FAIR_TAU * (1 - np.exp(-100 / (a * FAIR_TAU)))).sum() == pytest.approx(target, rel=1e-6)


def test_climate_reconstruction_against_observations():
    r = calibrate()
    assert r["co2_rmse_test_ppm"] < 5.0  # out-of-sample 2000-2023
    assert r["temp_rmse_test_k"] < 0.25
    assert 2.0 < r["ecs_k"] < 5.0


def test_zero_emissions_stabilise_concentration():
    d = historical_inputs()
    years = np.arange(1750, 2101)
    e = np.r_[d.emissions_gtco2.to_numpy(), np.zeros(2100 - 2023)]
    o = np.r_[d.other_forcing.to_numpy(), np.full(2100 - 2023, d.other_forcing.iloc[-1])]
    run = run_climate(years, e, o, None, ClimateParams(r0=28.5))
    assert run.co2_ppm[-1] < run.co2_ppm[years == 2023][0]  # sinks draw CO2 down after emissions stop
