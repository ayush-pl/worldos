"""Payment and credit services: the ONLY code that changes financial stocks.

Every function moves value between two balance sheets so that the accounting
identities in economy/state.py hold by construction; the bank's equity is
updated only through its income statement (interest, write-offs, dividends),
which is what makes the bank balance-sheet invariant a genuine check.

Settlement rules (single commercial bank + central bank):
* private <-> private payments move deposits inside the bank;
* government <-> private payments move deposits AND bank reserves, with the
  government's account at the central bank on the other side.
"""

from __future__ import annotations

import numpy as np


def _total(amounts: np.ndarray) -> float:
    if np.any(amounts < -1e-9):
        raise ValueError("payment amounts must be non-negative")
    return float(np.sum(amounts))


# --- private <-> private (deposit transfers) ----------------------------------------------
def households_pay_firms(world, hh: np.ndarray, firms: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.subtract.at(world.households.deposits, hh, amounts)
    np.add.at(world.firms.deposits, firms, amounts)
    return total


def firms_pay_households(world, firms: np.ndarray, hh: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.subtract.at(world.firms.deposits, firms, amounts)
    np.add.at(world.households.deposits, hh, amounts)
    np.add.at(world.households.income_cur, hh, amounts)
    return total


def distribute_to_owners(world, payer: str, amount: float, firm_amounts: np.ndarray | None = None) -> np.ndarray:
    """Pay ``amount`` to households pro rata to their equity shares.

    ``payer`` is "firms" (``firm_amounts`` per firm) or "bank" (paid out of bank equity).
    """
    h = world.households
    owners = np.flatnonzero(h.alive & (h.equity_share > 0))
    if amount <= 0 or not len(owners):
        return np.zeros(0)
    share = h.equity_share[owners] / h.equity_share[owners].sum()
    received = amount * share
    if payer == "firms":
        assert firm_amounts is not None and abs(firm_amounts.sum() - amount) <= 1e-6 * max(1.0, amount)
        world.firms.deposits -= firm_amounts
    elif payer == "bank":
        world.bank.equity -= amount
    else:
        raise ValueError(payer)
    h.deposits[owners] += received
    h.income_cur[owners] += received
    return received


# --- government <-> private (reserve settlement) -----------------------------------------
def government_pays_households(world, hh: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.add.at(world.households.deposits, hh, amounts)
    np.add.at(world.households.income_cur, hh, amounts)
    world.bank.reserves += total
    world.government.balance -= total
    return total


def households_pay_government(world, hh: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.subtract.at(world.households.deposits, hh, amounts)
    np.subtract.at(world.households.income_cur, hh, amounts)
    world.bank.reserves -= total
    world.government.balance += total
    return total


def government_pays_firms(world, firms: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.add.at(world.firms.deposits, firms, amounts)
    world.bank.reserves += total
    world.government.balance -= total
    return total


def firms_pay_government(world, firms: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.subtract.at(world.firms.deposits, firms, amounts)
    world.bank.reserves -= total
    world.government.balance += total
    return total


# --- bank <-> private ---------------------------------------------------------------------
def bank_pays_deposit_interest(world, monthly_rate: float) -> tuple[float, float]:
    h, f = world.households, world.firms
    ih = np.where(h.alive, np.maximum(h.deposits, 0.0) * monthly_rate, 0.0)
    jf = np.where(f.alive, np.maximum(f.deposits, 0.0) * monthly_rate, 0.0)
    h.deposits += ih
    h.income_cur += ih
    f.deposits += jf
    f.deposit_interest = jf
    total = float(ih.sum() + jf.sum())
    world.bank.equity -= total
    return float(ih.sum()), float(jf.sum())


def firms_pay_interest(world, firms: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.subtract.at(world.firms.deposits, firms, amounts)
    world.bank.equity += total
    return total


def lend_to_firms(world, firms: np.ndarray, amounts: np.ndarray) -> float:
    """Endogenous money: a loan creates a matching deposit."""
    total = _total(amounts)
    np.add.at(world.firms.loans, firms, amounts)
    np.add.at(world.firms.deposits, firms, amounts)
    return total


def firms_repay(world, firms: np.ndarray, amounts: np.ndarray) -> float:
    total = _total(amounts)
    np.subtract.at(world.firms.loans, firms, amounts)
    np.subtract.at(world.firms.deposits, firms, amounts)
    return total


def settle_central_bank_interest(world, monthly_rate: float) -> tuple[float, float]:
    """Central bank remunerates reserves and charges the government on its overdraft, both at the
    policy rate. Because reserves + government balance = 0, the central bank's net income is zero
    and its (zero) equity is preserved. Returns (interest to bank, interest paid by government)."""
    to_bank = world.bank.reserves * monthly_rate  # negative if the bank borrows from the CB
    by_gov = -world.government.balance * monthly_rate
    world.bank.reserves += to_bank
    world.bank.equity += to_bank
    world.government.balance -= by_gov
    return to_bank, by_gov


def write_off_firm_loans(world, firms: np.ndarray) -> float:
    total = float(world.firms.loans[firms].sum())
    world.firms.loans[firms] = 0.0
    world.bank.equity -= total
    return total
