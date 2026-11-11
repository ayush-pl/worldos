"""Engine contract.

An engine is a *stateless* transition function over the world state: all
persistent state lives in ``World`` (so cloning a world clones a complete,
resumable simulation). Engines declare when they run and in which order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class Engine(ABC):
    name: ClassVar[str]
    order: ClassVar[int]
    period_months: ClassVar[int] = 1
    phase: ClassVar[int] = 0

    @abstractmethod
    def step(self, world) -> None:  # noqa: ANN001 - World imported lazily to avoid cycles
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} every {self.period_months}m>"
