"""The authoritative world state.

``World`` holds *all* mutable simulation state, so ``World.clone()`` produces a
complete, independent, resumable copy - the primitive behind scenario
branching (docs/13 in the roadmap; worldos/scenario).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from worldos.config import WorldConfig
from worldos.core.actions import ActionKernel
from worldos.core.clock import EventScheduler, SimulationClock
from worldos.core.rng import RandomStreams
from worldos.economy.state import BankState, CentralBankState, FirmTable, Flows, GovernmentState
from worldos.geo.hierarchy import GeoHierarchy
from worldos.geo.synthetic import SettlementLayout
from worldos.population.schedules import FertilitySchedule, MortalitySchedule
from worldos.population.tables import HouseholdTable, PersonTable


@dataclass
class World:
    config: WorldConfig
    clock: SimulationClock
    rng: RandomStreams
    geo: GeoHierarchy
    layout: SettlementLayout
    mortality: MortalitySchedule
    fertility: FertilitySchedule
    persons: PersonTable
    households: HouseholdTable
    firms: FirmTable
    bank: BankState
    government: GovernmentState
    central_bank: CentralBankState
    kernel: ActionKernel
    events: EventScheduler
    flows: Flows = field(default_factory=Flows)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    strategic_agents: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

    def clone(self) -> "World":
        return copy.deepcopy(self)

    def begin_tick(self) -> None:
        self.flows.reset()
        self.households.income_cur = 0.0
        f = self.firms
        self.flows.inventory_start_units = float(f.inventory[f.alive].sum())
        f.distressed = False
