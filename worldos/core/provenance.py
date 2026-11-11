"""Provenance and epistemic status of every value in WORLD-OS.

The specification requires a strict separation between what was observed and
what was computed. Every observation stored by the data layer carries a
``Provenance`` record with its kind, source, reference date, spatial
resolution, unit, licence and uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ValueKind(str, Enum):
    OBSERVED = "observed"      # directly measured / published statistic
    ESTIMATED = "estimated"    # statistical estimate by a data producer (e.g. survey-based)
    INFERRED = "inferred"      # derived by WORLD-OS from observed data (e.g. downscaling)
    SIMULATED = "simulated"    # produced by a WORLD-OS simulation run
    FORECAST = "forecast"      # scenario projection beyond the last observation
    ASSUMED = "assumed"        # modelling assumption / illustrative parameter, not data


@dataclass(frozen=True)
class Uncertainty:
    """Either a standard deviation, a [low, high] interval with a coverage level, or 'unquantified'."""

    sd: float | None = None
    low: float | None = None
    high: float | None = None
    coverage: float | None = None  # e.g. 0.9 for a 90% interval
    note: str = ""

    @property
    def quantified(self) -> bool:
        return self.sd is not None or (self.low is not None and self.high is not None)


UNQUANTIFIED = Uncertainty(note="unquantified")


@dataclass(frozen=True)
class Provenance:
    kind: ValueKind
    source: str
    unit: str
    license: str
    reference_date: str | None = None
    geo_resolution: str | None = None
    uncertainty: Uncertainty = UNQUANTIFIED
    lineage: tuple[str, ...] = field(default_factory=tuple)  # upstream dataset ids / run ids
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("provenance.source is required")
        if not self.unit:
            raise ValueError("provenance.unit is required (use '1' for dimensionless)")
        if not self.license:
            raise ValueError("provenance.license is required (use 'unknown' only with a note)")
        if self.license == "unknown" and not self.notes:
            raise ValueError("an 'unknown' licence must be explained in notes")


def assumed(source: str, unit: str, notes: str = "") -> Provenance:
    """Provenance for an explicit modelling assumption authored in this repository."""
    return Provenance(ValueKind.ASSUMED, source=source, unit=unit, license="Apache-2.0 (this repository)", notes=notes)
