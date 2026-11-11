"""Construction of an accounting-consistent initial economy.

Initial stocks are set from flow targets (months of income/wage bill) and the
remaining items of the balance-sheet matrix are solved so every identity holds
at t = 0: bank reserves R = D + E - L and government balance G = -R.
"""

from __future__ import annotations

import numpy as np

from worldos.economy.engines import edu_productivity, effective_labour, mean_wage_per_worker
from worldos.economy.state import BankState, CentralBankState, FirmTable, GovernmentState
from worldos.population.tables import EMPLOYED, RETIRED, UNEMPLOYED

# Initial capital stock is a real asset held by firms (equity-financed); it does not enter the
# financial balance-sheet identities.


def initialize_economy(world, rng: np.random.Generator) -> None:
    ec, bc, gc, cc = world.config.economy, world.config.bank, world.config.government, world.config.central_bank
    p, h, lay = world.persons, world.households, world.layout
    n_f = ec.n_firms

    f = FirmTable(capacity=n_f)
    city = rng.choice(len(lay.city_ids), size=n_f, p=lay.city_share)
    size = rng.lognormal(0.0, ec.firm_size_sigma, size=n_f)
    size /= size.sum()
    W, A = ec.initial_wage, ec.initial_productivity
    # full-cost price fixed point: p = (1+mu) (W/A + delta * KY * p)
    mu, dky = ec.initial_markup, ec.depreciation_rate * ec.capital_output_ratio
    price0 = (1 + mu) * (W / A) / (1 - (1 + mu) * dky)
    f.append(n_f, fid=np.arange(n_f), city=city, region=lay.city_region[city], price=price0,
             capital_cost=dky * price0, productivity=A, wage_offer=W, markup=ec.initial_markup, loan_rate=cc.initial_policy_rate + bc.loan_spread)
    world.firms = f

    lf = np.flatnonzero(p.alive & (p.status == UNEMPLOYED))
    hired = lf[rng.random(len(lf)) >= ec.initial_unemployment]
    p.status[hired] = EMPLOYED
    p.employer[hired] = rng.choice(n_f, size=len(hired), p=size)

    L = effective_labour(world)
    out = A * L
    f.exp_demand = out
    f.normal_demand = out
    f.output_units = out
    f.sales_units = out
    f.sales_value = out * f.price
    f.inventory = ec.inventory_buffer * out
    f.capital = out * 12 * ec.capital_output_ratio  # at target utilisation
    f.invest_budget = ec.depreciation_rate / 12 * f.capital * price0
    f.target_output = out
    f.labour_demand = L
    bill = W * L
    f.deposits = ec.initial_firm_deposit_months * bill
    f.loans = ec.initial_firm_loan_months * bill

    # expected household disposable income under initial policy
    gov = GovernmentState(balance=0.0, income_tax_rate=gc.income_tax_rate, corporate_tax_rate=gc.corporate_tax_rate,
                          unemployment_replacement=gc.unemployment_replacement,
                          pension_replacement=gc.pension_replacement, spending_share=gc.spending_share)
    avg_wage = mean_wage_per_worker(world)
    emp = np.flatnonzero(p.alive & (p.status == EMPLOYED))
    wage_pp = W * edu_productivity(world)[p.edu[emp]]
    net = (wage_pp - gov.income_tax_rate * np.maximum(0.0, wage_pp - gc.tax_allowance_ratio * avg_wage)) * p.weight[emp]
    income = np.bincount(p.hh[emp], weights=net, minlength=len(h))
    for status, repl in ((UNEMPLOYED, gov.unemployment_replacement), (RETIRED, gov.pension_replacement)):
        idx = np.flatnonzero(p.alive & (p.status == status))
        income += np.bincount(p.hh[idx], weights=repl * avg_wage * p.weight[idx], minlength=len(h))
    # expected distributed profits (after corporate tax), allocated by equity share
    depreciation = ec.depreciation_rate / 12 * f.capital * price0
    profit = float((f.sales_value - bill - f.loans * f.loan_rate / 12 - depreciation).sum())
    income += ec.dividend_payout * (1 - gov.corporate_tax_rate) * max(profit, 0.0) * h.equity_share
    hetero = rng.lognormal(0.0, 0.5, size=len(h))
    # stationary deposits of the consumption function: (1 - c_y) * Yd = c_w * D  (scaled by heterogeneity)
    deposit_rate = max(0.0, cc.initial_policy_rate - bc.deposit_markdown) / 12
    months = ec.initial_household_deposit_months
    h.deposits = months * income * hetero / hetero.mean()
    income = income + deposit_rate * h.deposits
    h.income_exp = income
    h.income_last = income

    deposits = float(h.deposits.sum() + f.deposits.sum())
    loans = float(f.loans.sum() + h.loans.sum())
    equity = bc.initial_equity_ratio * loans
    reserves = deposits + equity - loans
    world.bank = BankState(reserves=reserves, equity=equity)
    gov.balance = -reserves
    world.government = gov
    world.central_bank = CentralBankState(policy_rate=cc.initial_policy_rate)
