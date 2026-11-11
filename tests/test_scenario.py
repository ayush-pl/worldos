import numpy as np
import pytest

from worldos.build import build_world, default_engines
from worldos.core.simulation import Simulation
from worldos.scenario.experiment import ExperimentSpec, Intervention, ScenarioSpec, render_markdown, run_experiment

SMALL = {"population.n_person_records": 3000, "economy.n_firms": 50}


def test_same_seed_is_reproducible(small_config):
    a = Simulation(build_world(small_config, 11)).run(30).metrics_frame()
    b = Simulation(build_world(small_config, 11)).run(30).metrics_frame()
    assert a.equals(b)


def test_clone_is_independent(small_config):
    w = build_world(small_config, 12)
    Simulation(w).run(6)
    c = w.clone()
    Simulation(c).run(6)
    assert w.clock.tick == 6 and c.clock.tick == 12
    assert len(w.metrics) == 6
    assert not np.shares_memory(w.households.deposits, c.households.deposits)


def test_identical_arms_have_zero_difference():
    spec = ExperimentSpec(name="placebo", base_overrides=SMALL, variants=[ScenarioSpec(name="placebo")],
                          seeds=[1, 2], warmup_months=6, horizon_months=12)
    res = run_experiment(spec)
    assert np.allclose(res.comparisons["diff_mean"].fillna(0), 0.0)


def test_intervention_is_applied_and_audited():
    spec = ExperimentSpec(
        name="tax", base_overrides=SMALL, seeds=[1], warmup_months=6, horizon_months=12,
        variants=[ScenarioSpec(name="spend", interventions=[
            Intervention(at_month=2, path="government.spending_share", value=0.2)])])
    res = run_experiment(spec)
    ts = res.timeseries
    assert set(ts.arm) == {"baseline", "spend"}
    md = render_markdown(res)
    assert "not forecasts" in md and "spend" in md


def test_disallowed_intervention_rejected():
    spec = ExperimentSpec(name="bad", base_overrides=SMALL, seeds=[1], warmup_months=1, horizon_months=3,
                          variants=[ScenarioSpec(name="x", interventions=[
                              Intervention(at_month=0, path="bank.equity", value=1e12)])])
    with pytest.raises(ValueError, match="rejected"):
        run_experiment(spec)


def test_spending_increase_lowers_unemployment():
    spec = ExperimentSpec(name="g", base_overrides={"population.n_person_records": 8000, "economy.n_firms": 100},
                          seeds=[1, 2, 3], warmup_months=12, horizon_months=36,
                          variants=[ScenarioSpec(name="more_g", overrides={"government.spending_share": 0.16})],
                          outcomes=["unemployment_rate", "gdp_real"])
    comp = run_experiment(spec).comparisons.set_index("outcome")
    assert comp.loc["unemployment_rate", "diff_mean"] < 0
    assert comp.loc["gdp_real", "diff_mean"] > 0
