"""Calibration of a real-country world (milestone M4, first step).

One free parameter, one target: the effective income-tax rate is chosen by bisection so that the
mean simulated unemployment rate over months [12, horizon) matches the observed (ILO-modelled,
WDI) rate in the base year, averaged across seeds. Unemployment falls monotonically as the tax
rate falls (more disposable income -> more demand) in this model, which makes bisection valid.
Everything else is either data, inferred from data, or a documented assumption; untargeted
moments are then compared with data in the country validation report.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from worldos.core.simulation import Simulation
from worldos.countries import sources
from worldos.countries.build import build_country_world


@dataclass
class CalibrationResult:
    iso3: str
    target_unemployment: float
    income_tax_rate: float
    achieved_unemployment: float
    converged: bool
    log: list[tuple[float, float]] = field(default_factory=list)


def simulate_unemployment(iso3: str, base_year: int, tax: float, seeds: list[int], months: int, n_records: int,
                          n_firms: int, rates) -> float:
    vals = []
    for seed in seeds:
        w = build_country_world(iso3, base_year, n_records, n_firms, seed,
                                overrides={"government.income_tax_rate": tax}, rates=rates, calibrated=False)
        df = Simulation(w).run(months).metrics_frame()
        vals.append(df.unemployment_rate.iloc[12:].mean())
    return float(np.mean(vals))


def calibrate_income_tax(iso3: str, base_year: int = 2023, seeds: list[int] | None = None, months: int = 60,
                         n_records: int = 10_000, n_firms: int = 150, bounds: tuple[float, float] = (0.0, 0.75),
                         tol: float = 0.003, max_iter: int = 12) -> CalibrationResult:
    seeds = seeds or [1, 2]
    target = sources.wdi_value(iso3, "unemployment_rate_ilo", base_year)
    if target is None:
        raise ValueError(f"no observed unemployment for {iso3}")
    target /= 100
    rates = sources.wpp_rates(iso3)
    lo, hi = bounds
    run = lambda t: simulate_unemployment(iso3, base_year, t, seeds, months, n_records, n_firms, rates)
    log = []
    u_lo, u_hi = run(lo), run(hi)
    log += [(lo, u_lo), (hi, u_hi)]
    if not (u_lo <= target <= u_hi):
        best = lo if abs(u_lo - target) < abs(u_hi - target) else hi
        return CalibrationResult(iso3, target, best, min(u_lo, u_hi, key=lambda u: abs(u - target)), False, log)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        u = run(mid)
        log.append((mid, u))
        if abs(u - target) <= tol:
            return CalibrationResult(iso3, target, mid, u, True, log)
        lo, hi = (mid, hi) if u < target else (lo, mid)
    mid, u = log[-1]
    return CalibrationResult(iso3, target, mid, u, abs(u - target) <= 2 * tol, log)


def save_calibration(result: CalibrationResult) -> str:
    path = sources.REAL_DATA.parent / "calibration" / f"{result.iso3}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2))
    return str(path)


def load_calibration(iso3: str) -> dict | None:
    path = sources.REAL_DATA.parent / "calibration" / f"{iso3}.json"
    return json.loads(path.read_text()) if path.exists() else None
