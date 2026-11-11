import numpy as np
import pytest

from worldos.core.simulation import Simulation
from worldos.countries.build import build_country_world
from worldos.energy.model import EnergyFit, SOURCES, backtest, mix_for_year


def _fit(rate=0.15, base_vre=0.05):
    shares = dict(zip(SOURCES, [0.4, 0.25, 0.1, 0.03, 0.05, base_vre * 0.6, base_vre * 0.4, 0.07, 0.05]))
    return EnergyFit("X", 2023, shares, 0.02, 0.6, rate, 2050.0)


def test_mix_sums_to_one_and_is_anchored_at_base_year():
    f = _fit()
    for year in (2023, 2030, 2050, 2100):
        m = mix_for_year(f, year)
        assert sum(m.values()) == pytest.approx(1.0)
        assert min(m.values()) >= 0
    m0 = mix_for_year(f, 2023)
    assert m0["wind"] + m0["solar"] == pytest.approx(0.05)


def test_faster_diffusion_means_more_renewables_later():
    slow, fast = mix_for_year(_fit(0.1), 2030), mix_for_year(_fit(0.2), 2030)
    assert fast["wind"] + fast["solar"] > slow["wind"] + slow["solar"]


def test_coal_phaseout_reaches_zero():
    f = _fit()
    assert mix_for_year(f, 2040, coal_phaseout_year=2040)["coal"] == pytest.approx(0.0)
    assert mix_for_year(f, 2030, coal_phaseout_year=2040)["coal"] < mix_for_year(f, 2030)["coal"]


@pytest.mark.parametrize("iso", ["IND", "USA", "DEU", "JPN"])
def test_energy_backtest_out_of_sample(iso):
    r = backtest(iso)
    assert 0.8 < r["reconciliation_k"] < 1.2  # IPCC factors explain fossil CO2
    assert r["energy_mape"] < 0.1
    assert r["co2_mape"] < 0.1


def test_country_energy_engine_runs():
    w = build_country_world("DEU", 2023, n_records=3000, n_firms=40)
    assert w.config.energy.enabled
    df = Simulation(w).run(24).metrics_frame()
    assert (df.primary_energy_twh_annualised > 0).all()
    assert df.wind_solar_share.iloc[-1] > df.wind_solar_share.iloc[0]
    assert sim_ok(df)


def sim_ok(df):
    return np.isfinite(df.co2_mt_annualised).all()
