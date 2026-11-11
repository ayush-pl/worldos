"""Columnar tables for person and household records.

A *record* represents ``weight`` identical real-world units (weighted
microsimulation). All monetary stocks and flows held on a household record
are TOTALS for the group it represents; per-household values are total/weight.
Within one run all records share the same weight (see docs/05).
"""

from __future__ import annotations

import numpy as np

from worldos.core.table import ColumnTable

# Person activity status codes
CHILD, STUDENT, EMPLOYED, UNEMPLOYED, INACTIVE, RETIRED = range(6)
STATUS_NAMES = ["child", "student", "employed", "unemployed", "inactive", "retired"]
LABOUR_FORCE = (EMPLOYED, UNEMPLOYED)
EXIT_NONE, EXIT_DEATH, EXIT_EMIGRATION = 0, 1, 2


class PersonTable(ColumnTable):
    schema = {
        "pid": (np.int64, -1),
        "hh": (np.int64, -1),
        "age_m": (np.int32, 0),
        "sex": (np.int8, 0),
        "edu": (np.int8, 0),  # attained/target education level 0..2
        "status": (np.int8, CHILD),
        "employer": (np.int32, -1),
        "alive": (np.bool_, True),
        "weight": (np.float64, 1.0),
        "birth_tick": (np.int32, 0),
        "exit_reason": (np.int8, 0),  # 0 present, 1 died, 2 emigrated
    }

    def __init__(self, capacity: int = 16) -> None:
        super().__init__(capacity)
        self.next_pid = 0

    def add(self, count: int, **values) -> np.ndarray:
        idx = self.append(count, **values)
        self._cols["pid"][idx] = np.arange(self.next_pid, self.next_pid + count)
        self.next_pid += count
        return idx


class HouseholdTable(ColumnTable):
    schema = {
        "hid": (np.int64, -1),
        "region": (np.int32, -1),
        "city": (np.int32, -1),
        "cell": (np.int32, -1),  # index into layout.cell_ids when a gridded layer exists
        "alive": (np.bool_, True),
        "weight": (np.float64, 1.0),
        "deposits": (np.float64, 0.0),
        "loans": (np.float64, 0.0),
        "equity_share": (np.float64, 0.0),  # share of firm + bank ownership
        "income_exp": (np.float64, 0.0),  # expected monthly disposable income (total for record)
        "income_cur": (np.float64, 0.0),  # disposable income accumulated in current month
        "income_last": (np.float64, 0.0),
        "consumption_last": (np.float64, 0.0),
        "n_members": (np.int32, 0),
    }

    def __init__(self, capacity: int = 16) -> None:
        super().__init__(capacity)
        self.next_hid = 0

    def add(self, count: int, **values) -> np.ndarray:
        idx = self.append(count, **values)
        self._cols["hid"][idx] = np.arange(self.next_hid, self.next_hid + count)
        self.next_hid += count
        return idx


def recount_members(persons: PersonTable, households: HouseholdTable) -> None:
    alive = persons.alive
    households.n_members = np.bincount(persons.hh[alive], minlength=len(households)).astype(np.int32)
