"""Central simulation clock and discrete-event scheduler.

The Alpha base tick is one calendar month. Engines declare a period in months
(1 = monthly, 3 = quarterly, 12 = annual) and a phase; the kernel runs every
due engine once per tick in a fixed, declared order. Sub-monthly processes
(weather, intraday markets) are planned as engine-internal sub-stepping, see
docs/03-system-architecture.md section "Time synchronization".
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True, order=True)
class SimDate:
    year: int
    month: int  # 1..12

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"month must be 1..12, got {self.month}")

    @classmethod
    def parse(cls, text: str) -> "SimDate":
        year, month = text.strip().split("-")[:2]
        return cls(int(year), int(month))

    @property
    def index(self) -> int:
        return self.year * 12 + (self.month - 1)

    def add_months(self, n: int) -> "SimDate":
        idx = self.index + n
        return SimDate(idx // 12, idx % 12 + 1)

    def months_since(self, other: "SimDate") -> int:
        return self.index - other.index

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


class SimulationClock:
    """Monotonic monthly clock. ``tick`` counts completed-or-running months from ``start``."""

    def __init__(self, start: SimDate) -> None:
        self.start = start
        self.tick = 0

    @property
    def now(self) -> SimDate:
        return self.start.add_months(self.tick)

    def advance(self) -> None:
        self.tick += 1

    def is_due(self, period_months: int, phase: int = 0) -> bool:
        if period_months <= 0:
            raise ValueError("period must be positive")
        return self.tick >= phase and (self.tick - phase) % period_months == 0

    def tick_of(self, date: SimDate) -> int:
        return date.months_since(self.start)


@dataclass(order=True)
class ScheduledEvent:
    tick: int
    priority: int
    seq: int
    name: str = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


class EventScheduler:
    """Priority queue of one-off events (scenario interventions, shocks).

    Events carry data, not code, so world snapshots stay serialisable; the
    kernel dispatches them to registered handlers by ``name``.
    """

    def __init__(self) -> None:
        self._heap: list[ScheduledEvent] = []
        self._seq = itertools.count()

    def schedule(self, tick: int, name: str, payload: dict[str, Any] | None = None, priority: int = 0) -> None:
        heapq.heappush(self._heap, ScheduledEvent(tick, priority, next(self._seq), name, payload or {}))

    def pop_due(self, tick: int) -> list[ScheduledEvent]:
        due = []
        while self._heap and self._heap[0].tick <= tick:
            due.append(heapq.heappop(self._heap))
        return due

    def __len__(self) -> int:
        return len(self._heap)

    def __deepcopy__(self, memo: dict) -> "EventScheduler":
        new = EventScheduler()
        new._heap = [ScheduledEvent(e.tick, e.priority, e.seq, e.name, dict(e.payload)) for e in self._heap]
        new._seq = itertools.count(max((e.seq for e in self._heap), default=-1) + 1)
        return new


EventHandler = Callable[[Any, ScheduledEvent], None]
