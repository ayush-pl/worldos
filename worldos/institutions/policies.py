"""Decision policies of strategic agents (agent class B).

A policy maps an *observation* (published indicators, with a reporting lag)
to a proposed ``Action``. It never touches world state. Alpha ships
transparent rule-based policies; optimisation-, RL- or LLM-based policies can
implement the same ``decide`` interface and remain subject to the kernel's
validation (docs/05-population-and-agents.md, section "Strategic agents").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from worldos.config import CentralBankConfig, GovernmentConfig
from worldos.core.actions import Action, DecisionSource


class Policy(Protocol):
    actor: str

    def decide(self, observation: dict[str, Any]) -> Action | None: ...


@dataclass
class TaylorRulePolicy:
    """Taylor (1993)-type rule with an unemployment gap in place of the output gap:

    i = r* + pi + phi_pi (pi - pi*) - phi_u (u - u*),  bounded below by zero.
    """

    cfg: CentralBankConfig
    actor: str = "central_bank"
    memory: list[dict[str, float]] = field(default_factory=list)

    def decide(self, obs: dict[str, Any]) -> Action | None:
        pi, u, current = obs.get("inflation_yoy"), obs.get("unemployment_rate"), obs["policy_rate"]
        if pi is None or u is None:
            return None
        c = self.cfg
        target = c.neutral_real_rate + pi + c.phi_inflation * (pi - c.inflation_target) \
            - c.phi_unemployment * (u - c.natural_unemployment)
        lo, hi = c.rate_bounds
        target = min(max(target, lo), hi)
        new = current + max(-c.max_step, min(c.max_step, target - current))  # gradualism
        self.memory.append({"pi": pi, "u": u, "target": target})
        return Action("set_policy_rate", self.actor, {"policy_rate": round(new, 6)}, DecisionSource.RULE,
                      rationale=f"Taylor rule: pi={pi:.4f}, u={u:.4f}, unconstrained target={target:.4f}")


@dataclass
class FiscalRulePolicy:
    """Debt-stabilising tax rule: adjust the income tax rate in proportion to the debt-ratio gap."""

    cfg: GovernmentConfig
    actor: str = "government"

    def decide(self, obs: dict[str, Any]) -> Action | None:
        ratio, current = obs.get("gov_debt_to_gdp"), obs["income_tax_rate"]
        if ratio is None:
            return None
        c = self.cfg
        delta = max(-c.max_tax_change, min(c.max_tax_change, c.fiscal_adjustment * (ratio - c.debt_target)))
        lo, hi = c.tax_rate_bounds
        new = min(max(current + delta, lo), hi)
        return Action("set_income_tax_rate", self.actor, {"income_tax_rate": round(new, 6)}, DecisionSource.RULE,
                      rationale=f"debt/GDP={ratio:.3f} vs target {c.debt_target:.3f}")
