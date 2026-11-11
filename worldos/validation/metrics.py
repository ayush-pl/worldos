"""Monthly indicator computation (the system's published statistics).

Definitions:
* GDP (expenditure) = C + I + G + change in inventories valued at full unit cost.
  GDP (income) = wages + operating surplus; both are reported and must agree.
* Real GDP: units produced (single-good economy; base-period prices = 1 unit).
* CPI: unit value of all final sales (Paasche-type for one good).
* Unemployment rate: unemployed / labour force (person weights).
* Poverty: share of persons in households with equivalised disposable income
  below 60% of the median (EU at-risk-of-poverty definition; OECD
  square-root equivalence scale).
* Wealth includes deposits, minus loans, plus the household's equity share of
  firm and bank net worth.
Well-being is deliberately reported as several indicators, not only GDP.
"""

from __future__ import annotations

import numpy as np

from worldos.core.engine import Engine
from worldos.economy.engines import capital_productivity, firm_net_worth, mean_wage_per_worker
from worldos.population.tables import EMPLOYED, RETIRED, UNEMPLOYED
from worldos.validation.stats import gini, weighted_quantile


class MetricsEngine(Engine):
    name = "metrics"
    order = 100

    def step(self, world) -> None:
        world.metrics.append(compute_metrics(world))


def compute_metrics(world) -> dict:
    p, h, f, fl = world.persons, world.households, world.firms, world.flows
    gov, bank = world.government, world.bank
    alive = p.alive
    w = p.weight
    pop = float(w[alive].sum())
    emp = float(w[alive & (p.status == EMPLOYED)].sum())
    unemp = float(w[alive & (p.status == UNEMPLOYED)].sum())
    lf = emp + unemp
    age_y = p.age_m / 12.0
    working_age = float(w[alive & (age_y >= 15) & (age_y < 65)].sum())
    old = float(w[alive & (age_y >= 65)].sum())

    gdp_exp = fl.consumption + fl.investment + fl.gov_purchases + fl.inventory_change_value
    op_surplus = fl.firm_revenue - fl.wages_paid + fl.inventory_change_value
    gdp_inc = fl.wages_paid + op_surplus
    prev = world.metrics
    cpi = fl.firm_revenue / fl.units_sold if fl.units_sold > 0 else (prev[-1]["cpi"] if prev else 1.0)
    infl = cpi / prev[-12]["cpi"] - 1.0 if len(prev) >= 12 else None
    annual_gdp = gdp_exp + sum(m["gdp_nominal"] for m in prev[-11:])
    annual_gdp *= 12.0 / (min(len(prev), 11) + 1)

    hh = np.flatnonzero(h.alive & (h.n_members > 0))
    size = h.n_members[hh].astype(float)
    eq_income = (h.income_last[hh] / h.weight[hh]) / np.sqrt(size)
    person_w = size * h.weight[hh]
    median_eq = weighted_quantile(eq_income, person_w, 0.5)
    alive_f = np.flatnonzero(f.alive)
    firm_nw = float(np.maximum(0.0, firm_net_worth(world, alive_f)).sum())
    capacity = capital_productivity(world) * f.capital[f.alive].sum()
    wealth = (h.deposits[hh] - h.loans[hh] + h.equity_share[hh] * (firm_nw + max(bank.equity, 0.0))) / h.weight[hh]
    loans = float(f.loans[f.alive].sum() + h.loans[h.alive].sum())

    return {
        "tick": world.clock.tick,
        "date": str(world.clock.now),
        "population": pop,
        "births": fl.births,
        "deaths": fl.deaths,
        "households": float(h.weight[hh].sum()),
        "old_age_dependency": old / working_age if working_age else float("nan"),
        "employed": emp,
        "unemployed": unemp,
        "labour_force": lf,
        "unemployment_rate": unemp / lf if lf else float("nan"),
        "employment_rate_15_64": emp / working_age if working_age else float("nan"),
        "mean_wage": mean_wage_per_worker(world),
        "gdp_nominal": gdp_exp,
        "gdp_income_side": gdp_inc,
        "gdp_real": fl.units_produced,
        "gdp_per_capita_annualised": 12 * gdp_exp / pop if pop else float("nan"),
        "consumption": fl.consumption,
        "investment": fl.investment,
        "investment_share_gdp": fl.investment / gdp_exp if gdp_exp else float("nan"),
        "consumption_share_gdp": fl.consumption / gdp_exp if gdp_exp else float("nan"),
        "capital_stock": float(f.capital[f.alive].sum()),
        "capacity_utilisation": fl.units_produced / capacity if capacity > 0 else float("nan"),
        "labour_share": fl.wages_paid / gdp_exp if gdp_exp else float("nan"),
        "gov_purchases": fl.gov_purchases,
        "inventory_change_value": fl.inventory_change_value,
        "wages": fl.wages_paid,
        "cpi": cpi,
        "inflation_yoy": infl,
        "policy_rate": world.central_bank.policy_rate,
        "income_tax_rate": gov.income_tax_rate,
        "gov_revenue": fl.income_tax + fl.corporate_tax,
        "gov_spending": fl.gov_purchases + fl.unemployment_benefits + fl.pensions + fl.gov_interest,
        "gov_interest": fl.gov_interest,
        "gov_debt": gov.debt,
        "gov_debt_to_gdp": gov.debt / annual_gdp if annual_gdp > 0 else float("nan"),
        "bank_equity": bank.equity,
        "bank_capital_ratio": bank.equity / loans if loans > 0 else float("nan"),
        "loans": loans,
        "new_loans": fl.new_loans,
        "credit_rationed": fl.credit_rationed,
        "household_deposits": float(h.deposits[h.alive].sum()),
        "firms_active": int(f.alive.sum()),
        "bankruptcies": fl.bankruptcies,
        "entries": fl.entries,
        "gini_income": gini(eq_income, person_w),
        "gini_wealth": gini(wealth, h.weight[hh]),
        "poverty_rate": float(person_w[eq_income < 0.6 * median_eq].sum() / person_w.sum()) if len(hh) else float("nan"),
        "median_equivalised_income": median_eq,
        "immigrants": fl.immigrants,
        "emigrants": fl.emigrants,
        "co2_mt_annualised": 12 * fl.co2_emissions_t / 1e6,
        "co2_intensity": world.emissions_intensity or 0.0,
        "primary_energy_twh_annualised": 12 * fl.primary_energy_mwh / 1e6,
        "wind_solar_share": (world.energy_mix["wind"] + world.energy_mix["solar"]) if world.energy_mix else float("nan"),
        "coal_share": world.energy_mix["coal"] if world.energy_mix else float("nan"),
        "retired_share": float(w[alive & (p.status == RETIRED)].sum()) / pop if pop else float("nan"),
    }
