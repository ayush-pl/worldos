"""Engines that run the observe -> decide -> validate -> apply cycle for strategic agents."""

from __future__ import annotations

from worldos.core.engine import Engine


def _observation(world) -> dict:
    """Agents observe the latest *published* indicators (one-month reporting lag)."""
    last = world.metrics[-1] if world.metrics else {}
    return {
        "inflation_yoy": last.get("inflation_yoy"),
        "unemployment_rate": last.get("unemployment_rate"),
        "gov_debt_to_gdp": last.get("gov_debt_to_gdp"),
        "policy_rate": world.central_bank.policy_rate,
        "income_tax_rate": world.government.income_tax_rate,
    }


class CentralBankEngine(Engine):
    name = "central_bank"
    order = 90
    period_months = 3
    phase = 2  # decides in the last month of each quarter

    def step(self, world) -> None:
        action = world.strategic_agents["central_bank"].decide(_observation(world))
        if action is not None:
            world.kernel.submit(action, world)


class GovernmentPolicyEngine(Engine):
    name = "government_policy"
    order = 95
    period_months = 12
    phase = 11  # annual budget decision in December

    def step(self, world) -> None:
        action = world.strategic_agents["government"].decide(_observation(world))
        if action is not None:
            world.kernel.submit(action, world)
