"""Simulation kernel: runs due engines in order each tick, dispatches events, checks invariants."""

from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd

from worldos.core.actions import Action, DecisionSource
from worldos.core.engine import Engine
from worldos.core.world import World
from worldos.validation.invariants import InvariantViolation, check_all


class Simulation:
    def __init__(self, world: World, engines: Iterable[Engine] | None = None, check_invariants: bool = True) -> None:
        if engines is None:
            from worldos.build import default_engines

            engines = default_engines()
        self.world = world
        self.engines = sorted(engines, key=lambda e: e.order)
        names = [e.name for e in self.engines]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate engine names: {names}")
        self.check_invariants = check_invariants
        self.violations: list[tuple[int, str]] = []

    def step(self) -> None:
        w = self.world
        w.begin_tick()
        for event in w.events.pop_due(w.clock.tick):
            if event.name == "scenario_intervention":
                result = w.kernel.submit(Action("scenario_intervention", "experimenter", event.payload,
                                                DecisionSource.SCENARIO), w)
                if not result.accepted:
                    raise ValueError(f"intervention rejected: {result.reasons}")
            else:
                raise ValueError(f"no handler for event {event.name!r}")
        for engine in self.engines:
            if w.clock.is_due(engine.period_months, engine.phase):
                engine.step(w)
        if self.check_invariants:
            problems = check_all(w)
            if problems:
                self.violations.extend((w.clock.tick, msg) for msg in problems)
                if w.config.simulation.strict_invariants:
                    raise InvariantViolation(f"tick {w.clock.tick} ({w.clock.now}): " + "; ".join(problems))
        w.clock.advance()

    def run(self, months: int, callback: Callable[[World], None] | None = None) -> "Simulation":
        for _ in range(months):
            self.step()
            if callback is not None:
                callback(self.world)
        return self

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.world.metrics)
