"""Provenance-tracked observation store and validated CSV ingestion.

Every stored value carries a ``Provenance`` record; ingestion validates the
schema, the unit against a registry, numeric finiteness and the dataset's
licence status, and records a content hash for lineage.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from worldos.core.provenance import Provenance, Uncertainty, ValueKind
from worldos.data.catalogue import DatasetEntry, ingestion_allowed

UNITS = {
    "1", "%", "persons", "persons/km2", "years", "births/woman", "deaths/1000", "USD", "USD/person",
    "constant 2017 USD", "local currency", "index", "km2", "ha", "t", "tCO2", "MtCO2", "GtCO2", "ppm",
    "degC", "K", "mm", "m3", "MWh", "TWh", "EJ",
}


@dataclass(frozen=True)
class Observation:
    entity_id: str  # geo id, e.g. "syn" or an ISO 3166 code
    variable: str
    time: str  # ISO date or year
    value: float
    provenance: Provenance


class ObservationStore:
    def __init__(self) -> None:
        self._obs: dict[tuple[str, str, str, str], Observation] = {}

    def add(self, obs: Observation) -> None:
        if obs.provenance.unit not in UNITS:
            raise ValueError(f"unit {obs.provenance.unit!r} not in the unit registry")
        if obs.value != obs.value or obs.value in (float("inf"), float("-inf")):
            raise ValueError("observation value must be finite")
        key = (obs.entity_id, obs.variable, obs.time, obs.provenance.source)
        if key in self._obs:
            raise ValueError(f"duplicate observation {key}")
        self._obs[key] = obs

    def query(self, entity_id: str, variable: str, kind: ValueKind | None = None) -> list[Observation]:
        out = [o for (e, v, _, _), o in self._obs.items() if e == entity_id and v == variable]
        if kind is not None:
            out = [o for o in out if o.provenance.kind == kind]
        return sorted(out, key=lambda o: o.time)

    def __len__(self) -> int:
        return len(self._obs)


REQUIRED_COLUMNS = ("entity_id", "variable", "time", "value")


def ingest_csv(
    store: ObservationStore,
    path: str | Path,
    dataset: DatasetEntry,
    unit: str,
    kind: ValueKind = ValueKind.OBSERVED,
    geo_resolution: str | None = None,
    licence_confirmed_by: str | None = None,
    sd_column: str | None = None,
) -> int:
    if not ingestion_allowed(dataset, licence_confirmed_by):
        raise PermissionError(f"licence of {dataset.id!r} must be confirmed by a named reviewer before ingestion")
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:16]
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    if not rows:
        return 0
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"missing columns {missing}")
    n = 0
    for i, row in enumerate(rows, start=2):
        try:
            value = float(row["value"])
        except ValueError as exc:
            raise ValueError(f"line {i}: non-numeric value {row['value']!r}") from exc
        unc = Uncertainty(sd=float(row[sd_column])) if sd_column and row.get(sd_column) else Uncertainty(note="unquantified")
        prov = Provenance(kind=kind, source=f"{dataset.id}:{Path(path).name}", unit=unit, license=dataset.license,
                          reference_date=row["time"], geo_resolution=geo_resolution, uncertainty=unc,
                          lineage=(f"sha256:{digest}",), notes=f"ingested from {dataset.provider}")
        store.add(Observation(row["entity_id"], row["variable"], row["time"], value, prov))
        n += 1
    return n
