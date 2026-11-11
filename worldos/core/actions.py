"""Typed actions, agent messages, and the validating action kernel.

Design rule (spec section 10): no agent - rule-based, optimiser, RL or LLM -
writes to authoritative world state. Agents *propose* typed ``Action``s; the
``ActionKernel`` validates them against registered constraints (legal bounds,
rate-of-change limits, budget feasibility) and only then applies them through
a registered service function. Every submission, accepted or rejected, is
written to an append-only audit log.

High-volume behavioural interactions (millions of purchases per month) are
not materialised as individual Python objects; they are exchanged as columnar
order batches inside market engines (see economy/markets.py), which enforce
the same accounting rules through the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


class DecisionSource(str, Enum):
    RULE = "rule"
    OPTIMIZER = "optimizer"
    LEARNED = "learned"
    LLM = "llm"
    HUMAN = "human"
    SCENARIO = "scenario"


class MessageType(str, Enum):
    PRODUCT_ORDER = "product_order"
    EMPLOYMENT_OFFER = "employment_offer"
    EMPLOYMENT_CONTRACT = "employment_contract"
    LOAN_APPLICATION = "loan_application"
    LOAN_DECISION = "loan_decision"
    INVESTMENT_OFFER = "investment_offer"
    NEGOTIATION = "negotiation"
    TRADE_AGREEMENT = "trade_agreement"
    POLICY_ANNOUNCEMENT = "policy_announcement"
    INSTITUTIONAL_DECISION = "institutional_decision"


@dataclass(frozen=True)
class Message:
    type: MessageType
    sender: str
    recipient: str  # agent id, or "*" for broadcast
    tick: int
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class Action:
    kind: str
    actor: str
    payload: Mapping[str, Any]
    source: DecisionSource = DecisionSource.RULE
    rationale: str = ""


@dataclass(frozen=True)
class AuditRecord:
    tick: int
    actor: str
    kind: str
    payload: Mapping[str, Any]
    source: DecisionSource
    accepted: bool
    reasons: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class ActionResult:
    accepted: bool
    reasons: tuple[str, ...] = ()


Validator = Callable[[Action, Any], list[str]]
Applier = Callable[[Action, Any], None]


@dataclass
class _Handler:
    validators: list[Validator]
    applier: Applier
    allowed_actors: frozenset[str]


class ActionKernel:
    def __init__(self) -> None:
        self._handlers: dict[str, _Handler] = {}
        self.audit: list[AuditRecord] = []
        self.messages: list[Message] = []

    def register(self, kind: str, applier: Applier, validators: list[Validator], allowed_actors: set[str]) -> None:
        self._handlers[kind] = _Handler(list(validators), applier, frozenset(allowed_actors))

    def submit(self, action: Action, world: Any) -> ActionResult:
        reasons: list[str] = []
        handler = self._handlers.get(action.kind)
        if handler is None:
            reasons.append(f"unknown action kind {action.kind!r}")
        else:
            if action.actor not in handler.allowed_actors:
                reasons.append(f"actor {action.actor!r} may not perform {action.kind!r}")
            for validator in handler.validators:
                reasons.extend(validator(action, world))
        accepted = not reasons
        if accepted:
            handler.applier(action, world)  # type: ignore[union-attr]
        tick = world.clock.tick
        self.audit.append(
            AuditRecord(tick, action.actor, action.kind, dict(action.payload), action.source, accepted, tuple(reasons), action.rationale)
        )
        if accepted:
            self.messages.append(
                Message(MessageType.POLICY_ANNOUNCEMENT, action.actor, "*", tick, {"kind": action.kind, **action.payload})
            )
        return ActionResult(accepted, tuple(reasons))
