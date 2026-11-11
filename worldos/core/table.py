"""Columnar (struct-of-arrays) entity tables.

Behavioural agents (people, households, firms) are stored column-wise in
NumPy arrays so that their decision rules can be vectorised. This is what
lets Alpha simulate 10^5-10^6 records on one core, and it is the layout the
distributed architecture partitions (docs/03-system-architecture.md).
Rows are never deleted; an ``alive`` column marks retired rows so that row
indices remain stable identifiers within a run.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np


class ColumnTable:
    schema: ClassVar[dict[str, tuple[Any, Any]]] = {}

    def __init__(self, capacity: int = 16) -> None:
        self.__dict__["n"] = 0
        self.__dict__["_cap"] = max(int(capacity), 1)
        self.__dict__["_cols"] = {
            name: np.full(self._cap, default, dtype=dtype) for name, (dtype, default) in self.schema.items()
        }

    def __getattr__(self, name: str) -> np.ndarray:
        cols = self.__dict__.get("_cols")
        if cols is not None and name in cols:
            return cols[name][: self.__dict__["n"]]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        cols = self.__dict__.get("_cols")
        if cols is not None and name in cols:
            cols[name][: self.n] = value
        else:
            self.__dict__[name] = value

    def __len__(self) -> int:
        return self.n

    @property
    def columns(self) -> list[str]:
        return list(self._cols)

    def _grow(self, needed: int) -> None:
        new_cap = max(needed, 2 * self._cap)
        for name, (dtype, default) in self.schema.items():
            arr = np.full(new_cap, default, dtype=dtype)
            arr[: self.n] = self._cols[name][: self.n]
            self._cols[name] = arr
        self.__dict__["_cap"] = new_cap

    def append(self, count: int, **values: Any) -> np.ndarray:
        """Append ``count`` rows; unspecified columns take schema defaults. Returns new row indices."""
        unknown = set(values) - set(self._cols)
        if unknown:
            raise KeyError(f"unknown columns: {sorted(unknown)}")
        start = self.n
        end = start + int(count)
        if end > self._cap:
            self._grow(end)
        for name, (dtype, default) in self.schema.items():
            self._cols[name][start:end] = values.get(name, default)
        self.__dict__["n"] = end
        return np.arange(start, end)

    def to_frame(self):  # pragma: no cover - convenience
        import pandas as pd

        return pd.DataFrame({k: v[: self.n] for k, v in self._cols.items()})
