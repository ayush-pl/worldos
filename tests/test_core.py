import copy

import numpy as np
import pytest

from worldos.core.actions import Action, ActionKernel
from worldos.core.clock import EventScheduler, SimDate, SimulationClock
from worldos.core.provenance import Provenance, ValueKind
from worldos.core.rng import RandomStreams
from worldos.core.table import ColumnTable


def test_simdate_arithmetic():
    d = SimDate(2025, 11)
    assert d.add_months(3) == SimDate(2026, 2)
    assert d.add_months(-11) == SimDate(2024, 12)
    assert SimDate.parse("2030-01").months_since(SimDate(2025, 1)) == 60
    with pytest.raises(ValueError):
        SimDate(2025, 13)


def test_clock_periods_and_phases():
    clock = SimulationClock(SimDate(2025, 1))
    due = []
    for _ in range(24):
        if clock.is_due(12, 11):
            due.append(str(clock.now))
        clock.advance()
    assert due == ["2025-12", "2026-12"]


def test_event_scheduler_orders_and_copies():
    s = EventScheduler()
    s.schedule(5, "b", priority=1)
    s.schedule(5, "a", priority=0)
    s.schedule(2, "c")
    clone = copy.deepcopy(s)
    assert [e.name for e in s.pop_due(5)] == ["c", "a", "b"]
    assert len(clone) == 3


def test_random_streams_are_independent_of_creation_order():
    a, b = RandomStreams(7), RandomStreams(7)
    a.get("x")
    x_after = a.get("y").random(3)
    y_direct = b.get("y").random(3)
    assert np.array_equal(x_after, y_direct)


def test_column_table_growth_and_defaults():
    class T(ColumnTable):
        schema = {"a": (np.float64, 1.5), "b": (np.int32, -1)}

    t = T(capacity=2)
    t.append(3, a=[1, 2, 3])
    t.append(5)
    assert len(t) == 8 and t.a[-1] == 1.5 and t.b[0] == -1
    t.a[0] = 9.0  # views are writable
    assert t.a[0] == 9.0
    with pytest.raises(KeyError):
        t.append(1, c=1)


def test_provenance_requires_core_fields():
    with pytest.raises(ValueError):
        Provenance(ValueKind.OBSERVED, source="", unit="1", license="CC BY 4.0")
    with pytest.raises(ValueError):
        Provenance(ValueKind.OBSERVED, source="x", unit="1", license="unknown")


class _W:
    class clock:
        tick = 0


def test_kernel_validates_audits_and_rejects():
    kernel = ActionKernel()
    applied = []
    kernel.register("set_x", lambda a, w: applied.append(a.payload["x"]),
                    [lambda a, w: [] if a.payload["x"] < 10 else ["too big"]], {"gov"})
    assert kernel.submit(Action("set_x", "gov", {"x": 3}), _W).accepted
    assert not kernel.submit(Action("set_x", "gov", {"x": 30}), _W).accepted
    assert not kernel.submit(Action("set_x", "bank", {"x": 1}), _W).accepted
    assert not kernel.submit(Action("unknown", "gov", {}), _W).accepted
    assert applied == [3]
    assert [r.accepted for r in kernel.audit] == [True, False, False, False]
