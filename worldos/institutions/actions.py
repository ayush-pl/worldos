"""Kernel registrations: which actions exist, who may take them, and their constraints."""

from __future__ import annotations

from worldos.config import get_path, set_path
from worldos.core.actions import Action, ActionKernel

ALLOWED_INTERVENTION_PREFIXES = ("config.", "government.", "central_bank.")
ALLOWED_GOVERNMENT_INSTRUMENTS = {
    "income_tax_rate", "corporate_tax_rate", "unemployment_replacement", "pension_replacement", "spending_share",
}


def _num(key: str):
    def check(action: Action, world) -> list[str]:
        v = action.payload.get(key)
        return [] if isinstance(v, (int, float)) and v == v else [f"{key} must be a finite number"]
    return check


def _policy_rate_bounds(action: Action, world) -> list[str]:
    c = world.config.central_bank
    v = action.payload.get("policy_rate", float("nan"))
    errs = []
    if not c.rate_bounds[0] <= v <= c.rate_bounds[1]:
        errs.append(f"policy rate {v} outside legal bounds {c.rate_bounds}")
    if abs(v - world.central_bank.policy_rate) > c.max_step + 1e-12:
        errs.append(f"policy rate change exceeds max step {c.max_step}")
    return errs


def _tax_bounds(action: Action, world) -> list[str]:
    c = world.config.government
    v = action.payload.get("income_tax_rate", float("nan"))
    errs = []
    if not c.tax_rate_bounds[0] <= v <= c.tax_rate_bounds[1]:
        errs.append(f"income tax rate {v} outside bounds {c.tax_rate_bounds}")
    if abs(v - world.government.income_tax_rate) > c.max_tax_change + 1e-12:
        errs.append(f"income tax change exceeds {c.max_tax_change} per decision")
    return errs


def _intervention_path(action: Action, world) -> list[str]:
    path = action.payload.get("path", "")
    if not path.startswith(ALLOWED_INTERVENTION_PREFIXES):
        return [f"intervention path {path!r} not in an allowed namespace {ALLOWED_INTERVENTION_PREFIXES}"]
    try:
        get_path(world, path)
    except AttributeError:
        return [f"unknown intervention path {path!r}"]
    if path.startswith("government.") and path.split(".", 1)[1] not in ALLOWED_GOVERNMENT_INSTRUMENTS:
        return [f"{path!r} is not a government policy instrument"]
    return []


def _apply_policy_rate(action: Action, world) -> None:
    world.central_bank.policy_rate = float(action.payload["policy_rate"])


def _apply_tax(action: Action, world) -> None:
    world.government.income_tax_rate = float(action.payload["income_tax_rate"])


def _apply_intervention(action: Action, world) -> None:
    set_path(world, action.payload["path"], action.payload["value"])


def register_default_actions(kernel: ActionKernel) -> None:
    kernel.register("set_policy_rate", _apply_policy_rate, [_num("policy_rate"), _policy_rate_bounds], {"central_bank"})
    kernel.register("set_income_tax_rate", _apply_tax, [_num("income_tax_rate"), _tax_bounds], {"government"})
    kernel.register("scenario_intervention", _apply_intervention, [_intervention_path], {"experimenter"})
