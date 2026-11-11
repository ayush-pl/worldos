"""Accounting and consistency invariants, checked after every tick.

These are necessary (not sufficient) conditions for a correct simulation:
they detect money or goods created or destroyed outside the ledger, broken
cross-references between entities, and impossible states.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from worldos.population.tables import EMPLOYED, RETIRED


class InvariantViolation(AssertionError):
    pass


def _close(a: float, b: float, scale: float, rel: float) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(scale))


def bank_balance_sheet(world, rel: float) -> list[str]:
    f, h, b = world.firms, world.households, world.bank
    deposits = float(f.deposits.sum() + h.deposits.sum())
    loans = float(f.loans.sum() + h.loans.sum())
    residual = b.reserves + loans - deposits
    if not _close(residual, b.equity, deposits + loans, rel):
        return [f"bank: reserves + loans - deposits = {residual:.6f} != tracked equity {b.equity:.6f}"]
    return []


def central_bank_balance_sheet(world, rel: float) -> list[str]:
    total = world.bank.reserves + world.government.balance
    if not _close(total, 0.0, abs(world.bank.reserves) + abs(world.government.balance), rel):
        return [f"central bank: reserves + government account = {total:.6f} != 0"]
    return []


def goods_conservation(world, rel: float) -> list[str]:
    fl, f = world.flows, world.firms
    end = float(f.inventory[f.alive].sum())
    expected = fl.inventory_start_units + fl.units_produced - fl.units_sold - fl.units_written_off
    if not _close(end, expected, fl.units_produced + fl.inventory_start_units, rel * 1e3):
        return [f"goods: closing inventory {end:.6f} != opening + produced - sold - written off {expected:.6f}"]
    return []


def capital_conservation(world, rel: float) -> list[str]:
    fl, f = world.flows, world.firms
    end = float(f.capital.sum())
    expected = fl.capital_start_units + fl.investment_units - fl.depreciation_units
    if not _close(end, expected, fl.capital_start_units, rel * 1e3):
        return [f"capital: closing stock {end:.6f} != opening + investment - depreciation {expected:.6f}"]
    if (f.capital < -1e-9).any():
        return ["negative capital stock"]
    return []


def expenditure_equals_revenue(world, rel: float) -> list[str]:
    fl = world.flows
    errs = []
    final = fl.consumption + fl.investment + fl.gov_purchases
    if not _close(final, fl.firm_revenue, fl.firm_revenue, rel):
        errs.append(f"GDP identity: C + I + G = {final:.6f} != firm revenue {fl.firm_revenue:.6f}")
    if not _close(fl.wages_paid, fl.wages_received, fl.wages_paid, rel):
        errs.append(f"wages paid {fl.wages_paid:.6f} != wages received {fl.wages_received:.6f}")
    return errs


def feasibility(world, rel: float) -> list[str]:
    h, f = world.households, world.firms
    errs = []
    scale = 1e-7 * max(1.0, float(np.abs(h.deposits).max(initial=0.0)))
    if (h.deposits[h.alive] < -scale).any():
        errs.append("household deposits negative (overdrafts are not modelled in Alpha)")
    if (f.deposits[f.alive] < -1e-7 * max(1.0, float(np.abs(f.deposits).max(initial=0.0)))).any():
        errs.append("firm deposits negative")
    if (f.inventory < -1e-9).any() or (f.loans < -1e-6).any():
        errs.append("negative inventory or loans")
    if (f.price[f.alive] <= 0).any() or (f.wage_offer[f.alive] <= 0).any():
        errs.append("non-positive price or wage")
    if not np.isfinite([world.bank.equity, world.bank.reserves, world.government.balance]).all():
        errs.append("non-finite aggregate balance")
    return errs


def population_consistency(world, rel: float) -> list[str]:
    p, h, f = world.persons, world.households, world.firms
    errs = []
    alive = p.alive
    emp = alive & (p.status == EMPLOYED)
    if (p.employer[emp] < 0).any() or (~f.alive[p.employer[emp]]).any():
        errs.append("employed person without a living employer")
    if (p.employer[alive & ~emp] >= 0).any():
        errs.append("non-employed person attached to an employer")
    if (~h.alive[p.hh[alive]]).any():
        errs.append("living person in a dissolved household")
    if (p.weight[alive] <= 0).any():
        errs.append("non-positive person weight")
    if ((p.status == RETIRED) & alive & (p.age_m < world.config.population.pension_age * 12)).any():
        errs.append("retired person below pension age")
    share = h.equity_share[h.alive].sum()
    if abs(share - 1.0) > 1e-6:
        errs.append(f"household equity shares sum to {share:.8f}, not 1")
    return errs


INVARIANTS: dict[str, Callable] = {
    "bank_balance_sheet": bank_balance_sheet,
    "central_bank_balance_sheet": central_bank_balance_sheet,
    "goods_conservation": goods_conservation,
    "capital_conservation": capital_conservation,
    "expenditure_equals_revenue": expenditure_equals_revenue,
    "feasibility": feasibility,
    "population_consistency": population_consistency,
}


def check_all(world) -> list[str]:
    rel = world.config.simulation.invariant_rel_tol
    out: list[str] = []
    for name, fn in INVARIANTS.items():
        out.extend(f"[{name}] {msg}" for msg in fn(world, rel))
    return out
