"""Economic state: firm table, bank, government, central bank, per-tick flows.

Balance-sheet matrix of the Alpha economy (single bank, closed economy):

                    Households  Firms   Bank     Government  Central bank   Sum
  Deposits             +Dh       +Df    -D                                  0
  Loans                -Lh       -Lf    +L                                  0
  Reserves (net)                        +R                      -R          0
  Gov. account (net)                                 +G         -G          0
  Inventories, capital           +pK                                      +pK
  Net worth            NWh       NWf    E           G            0       pK

with D = Dh + Df, L = Lh + Lf, and the central-bank identity R + G = 0
(the central bank has zero equity in Alpha). The bank's equity E is tracked
from its income statement and must equal R + L - D at every tick
(checked in worldos/validation/invariants.py).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from worldos.core.table import ColumnTable


class FirmTable(ColumnTable):
    schema = {
        "fid": (np.int64, -1),
        "city": (np.int32, -1),
        "region": (np.int32, -1),
        "alive": (np.bool_, True),
        "price": (np.float64, 1.0),
        "markup": (np.float64, 0.1),
        "productivity": (np.float64, 1.0),  # goods per effective worker-month
        "wage_offer": (np.float64, 1.0),  # SYN$ per effective worker-month
        "inventory": (np.float64, 0.0),  # goods
        "capital": (np.float64, 0.0),  # installed capital, goods
        "capital_cost": (np.float64, 0.0),  # depreciation cost per unit of output at normal utilisation
        "invest_budget": (np.float64, 0.0),  # planned investment spending this month
        "investment_units": (np.float64, 0.0),
        "investment_value": (np.float64, 0.0),
        "depreciation_value": (np.float64, 0.0),
        "exp_demand": (np.float64, 0.0),  # goods / month
        "normal_demand": (np.float64, 0.0),  # slow-moving demand used for capacity planning
        "target_output": (np.float64, 0.0),
        "labour_demand": (np.float64, 0.0),  # effective workers
        "unfilled": (np.float64, 0.0),  # effective vacancies left unfilled this month
        "fired": (np.bool_, False),
        "deposits": (np.float64, 0.0),
        "loans": (np.float64, 0.0),
        "loan_rate": (np.float64, 0.0),  # annual
        "output_units": (np.float64, 0.0),
        "sales_units": (np.float64, 0.0),
        "sales_value": (np.float64, 0.0),
        "wage_bill": (np.float64, 0.0),
        "interest_paid": (np.float64, 0.0),
        "deposit_interest": (np.float64, 0.0),
        "profit": (np.float64, 0.0),
        "distressed": (np.bool_, False),
        "dead_months": (np.int32, 0),
        "age_m": (np.int32, 0),
    }

    def unit_cost(self) -> np.ndarray:
        """Full unit cost: labour cost plus depreciation of capital at normal utilisation."""
        return self.wage_offer / self.productivity + self.capital_cost


@dataclass
class BankState:
    reserves: float = 0.0  # net position at the central bank (may be negative = CB credit)
    equity: float = 0.0  # tracked from the income statement


@dataclass
class GovernmentState:
    balance: float = 0.0  # net position at the central bank; negative = debt
    income_tax_rate: float = 0.0
    corporate_tax_rate: float = 0.0
    unemployment_replacement: float = 0.0
    pension_replacement: float = 0.0
    spending_share: float = 0.0

    @property
    def debt(self) -> float:
        return -self.balance


@dataclass
class CentralBankState:
    policy_rate: float = 0.0


@dataclass
class Flows:
    """Flows recorded during the current tick (reset at the start of every tick)."""

    wages_paid: float = 0.0  # by firms
    wages_received: float = 0.0  # by households (gross)
    income_tax: float = 0.0
    unemployment_benefits: float = 0.0
    pensions: float = 0.0
    consumption: float = 0.0  # household spending
    consumption_units: float = 0.0
    investment: float = 0.0
    investment_units: float = 0.0
    depreciation_units: float = 0.0
    depreciation_value: float = 0.0
    capital_start_units: float = 0.0
    gov_purchases: float = 0.0
    gov_purchase_units: float = 0.0
    firm_revenue: float = 0.0
    units_produced: float = 0.0
    units_sold: float = 0.0
    units_written_off: float = 0.0
    inventory_start_units: float = 0.0
    inventory_change_value: float = 0.0  # (produced - sold) valued at unit cost
    loan_interest: float = 0.0
    reserve_interest: float = 0.0
    gov_interest: float = 0.0
    working_capital_loans: float = 0.0
    deposit_interest: float = 0.0
    new_loans: float = 0.0
    repayments: float = 0.0
    write_offs: float = 0.0
    firm_dividends: float = 0.0
    bank_dividends: float = 0.0
    corporate_tax: float = 0.0
    births: float = 0.0  # represented persons
    deaths: float = 0.0
    immigrants: float = 0.0
    co2_emissions_t: float = 0.0
    primary_energy_mwh: float = 0.0
    emigrants: float = 0.0
    bankruptcies: int = 0
    entries: int = 0
    hires: float = 0.0
    separations: float = 0.0
    credit_rationed: float = 0.0

    def reset(self) -> None:
        for f in fields(self):
            setattr(self, f.name, type(getattr(self, f.name))(0))
