import numpy as np
import pytest

from worldos.build import build_world
from worldos.core.simulation import Simulation
from worldos.countries.calibrate import load_calibration
from worldos.economy.engines import capital_productivity
from worldos.validation.invariants import check_all


def test_capital_cannot_appear_from_nowhere(small_config):
    w = build_world(small_config, 1)
    sim = Simulation(w)
    sim.step()
    bump = 0.01 * float(w.firms.capital.sum())
    w.flows.reset()
    w.flows.capital_start_units = float(w.firms.capital.sum())
    w.firms.capital[0] += bump
    assert any("capital" in m for m in check_all(w))


def test_production_respects_capacity(small_config):
    w = build_world(small_config, 2)
    Simulation(w).run(12)
    f = w.firms
    assert np.all(f.output_units <= capital_productivity(w) * (f.capital / (1 - w.config.economy.depreciation_rate / 12)) + 1e-6)


def test_investment_and_shares_plausible(small_config):
    df = Simulation(build_world(small_config, 3)).run(72).metrics_frame().iloc[24:]
    assert 0.12 < df.investment_share_gdp.mean() < 0.32
    assert 0.5 < df.consumption_share_gdp.mean() < 0.8
    assert 0.4 < df.labour_share.mean() < 0.85
    assert np.allclose(df.consumption + df.investment + df.gov_purchases + df.inventory_change_value, df.gdp_nominal)


def test_country_calibrations_targets_met():
    for iso in ["IND", "USA", "DEU", "JPN", "BRA", "NGA"]:
        c = load_calibration(iso)
        assert c["converged"]
        assert abs(c["achieved_unemployment"] - c["target_unemployment"]) < 0.006
        if c["target_investment_share"] is not None:
            assert abs(c["achieved_investment_share"] - c["target_investment_share"]) < 0.01
