import numpy as np
import pytest

from worldos.build import build_world, default_engines
from worldos.core.simulation import Simulation
from worldos.economy import ledger
from worldos.validation.invariants import InvariantViolation, check_all


def test_initial_state_satisfies_all_invariants(small_config):
    w = build_world(small_config, 1)
    w.begin_tick()
    assert check_all(w) == []


def test_long_run_keeps_invariants_and_plausible_ranges(small_config):
    w = build_world(small_config, 2)
    sim = Simulation(w, default_engines()).run(120)  # strict mode raises on any violation
    df = sim.metrics_frame()
    assert sim.violations == []
    late = df.iloc[60:]
    assert late.unemployment_rate.between(0.0, 0.3).all()
    assert late.cpi.gt(0).all()
    assert (df.gdp_nominal - df.gdp_income_side).abs().max() < 1e-6 * df.gdp_nominal.max()
    assert late.firms_active.min() > 0.5 * small_config.economy.n_firms


def test_ledger_detects_one_sided_money_creation(small_config):
    w = build_world(small_config, 3)
    w.begin_tick()
    w.households.deposits[0] += 1_000.0  # money appearing from nowhere
    problems = check_all(w)
    assert any("bank" in msg for msg in problems)


def test_loans_create_deposits_and_writeoffs_hit_bank_equity(small_config):
    w = build_world(small_config, 3)
    w.begin_tick()
    e0 = w.bank.equity
    ledger.lend_to_firms(w, np.array([0]), np.array([5_000.0]))
    assert check_all(w) == []
    ledger.write_off_firm_loans(w, np.array([0]))
    assert w.bank.equity < e0
    assert check_all(w) == []


def test_government_payments_settle_in_reserves(small_config):
    w = build_world(small_config, 3)
    w.begin_tick()
    r0, g0 = w.bank.reserves, w.government.balance
    ledger.government_pays_households(w, np.array([0, 1]), np.array([10.0, 20.0]))
    assert w.bank.reserves - r0 == pytest.approx(30.0)
    assert g0 - w.government.balance == pytest.approx(30.0)
    assert check_all(w) == []


def test_strict_mode_raises(small_config):
    w = build_world(small_config, 6)
    sim = Simulation(w, default_engines())
    sim.step()
    w.firms.deposits[0] += 1e6
    with pytest.raises(InvariantViolation):
        sim.step()


def test_policy_agents_act_through_kernel(small_config):
    w = build_world(small_config, 7)
    Simulation(w, default_engines()).run(24)
    kinds = {r.kind for r in w.kernel.audit}
    assert {"set_policy_rate", "set_income_tax_rate"} <= kinds
    assert all(r.accepted for r in w.kernel.audit)
    # an out-of-bounds proposal is rejected and leaves state unchanged
    from worldos.core.actions import Action, DecisionSource

    before = w.central_bank.policy_rate
    res = w.kernel.submit(Action("set_policy_rate", "central_bank", {"policy_rate": 0.5}, DecisionSource.LLM), w)
    assert not res.accepted and w.central_bank.policy_rate == before
    res = w.kernel.submit(Action("set_income_tax_rate", "central_bank", {"income_tax_rate": 0.3}), w)
    assert not res.accepted
