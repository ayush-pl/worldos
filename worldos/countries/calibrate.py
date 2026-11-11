"""Calibration of real-country worlds (milestone M4).

Two parameters, two targets, solved by alternating one-dimensional bisections (Gauss-Seidel):

* effective income-tax rate      -> unemployment rate (ILO-modelled, WDI)       [unemployment rises with tax]
* capital-output ratio           -> investment share of GDP (WDI gross capital  [investment share rises with K/Y]
                                    formation, which also includes public and housing investment that the
                                    model does not separate: an approximation)

Simulated moments are means over months [12, horizon) and across seeds. If a country lacks the investment
target, only the tax rate is calibrated. Every other parameter is data, inferred from data, or a documented
assumption; untargeted moments are compared with data in the country validation report.
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
    target_investment_share: float | None = None
    capital_output_ratio: float | None = None
    achieved_investment_share: float | None = None
    log: list[tuple] = field(default_factory=list)


def simulate_moments(iso3: str, base_year: int, overrides: dict, seeds: list[int], months: int, n_records: int,
                     n_firms: int, rates) -> tuple[float, float]:
    u, inv = [], []
    for seed in seeds:
        w = build_country_world(iso3, base_year, n_records, n_firms, seed, overrides=overrides, rates=rates,
                                calibrated=False)
        df = Simulation(w).run(months).metrics_frame().iloc[12:]
        u.append(df.unemployment_rate.mean())
        inv.append((df.investment / df.gdp_nominal).mean())
    return float(np.mean(u)), float(np.mean(inv))


def _bisect(f, target: float, lo: float, hi: float, increasing: bool, tol: float, max_iter: int, log: list, name: str):
    """Find x with f(x) ~ target for a monotone f; returns (x, f(x), converged)."""
    f_lo, f_hi = f(lo), f(hi)
    log += [(name, lo, f_lo), (name, hi, f_hi)]
    lo_v, hi_v = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    if not (lo_v <= target <= hi_v):
        x, v = min(((lo, f_lo), (hi, f_hi)), key=lambda t: abs(t[1] - target))
        return x, v, False
    x, v = lo, f_lo
    for _ in range(max_iter):
        x = 0.5 * (lo + hi)
        v = f(x)
        log.append((name, x, v))
        if abs(v - target) <= tol:
            return x, v, True
        if (v < target) == increasing:
            lo = x
        else:
            hi = x
    return x, v, abs(v - target) <= 2 * tol


def calibrate_country(iso3: str, base_year: int = 2023, seeds: list[int] | None = None, months: int = 60,
                      n_records: int = 10_000, n_firms: int = 150, rounds: int = 3) -> CalibrationResult:
    seeds = seeds or [1, 2]
    u_target = sources.wdi_value(iso3, "unemployment_rate_ilo", base_year)
    if u_target is None:
        raise ValueError(f"no observed unemployment for {iso3}")
    u_target /= 100
    i_target = sources.wdi_value(iso3, "gross_capital_formation_share_gdp", base_year)
    i_target = None if i_target is None else i_target / 100
    rates = sources.wpp_rates(iso3)
    params = {"government.income_tax_rate": 0.4, "economy.capital_output_ratio": 3.0}
    log: list = []
    cache: dict = {}

    def moments(p: dict) -> tuple[float, float]:
        key = tuple(round(v, 6) for v in p.values())
        if key not in cache:
            cache[key] = simulate_moments(iso3, base_year, dict(p), seeds, months, n_records, n_firms, rates)
        return cache[key]

    ok_u = ok_i = False
    for _ in range(rounds):
        params["government.income_tax_rate"], u, ok_u = _bisect(
            lambda t: moments({**params, "government.income_tax_rate": t})[0], u_target, 0.0, 0.85, True,
            0.003, 12, log, "tax")
        if i_target is None:
            break
        params["economy.capital_output_ratio"], inv, ok_i = _bisect(
            lambda k: moments({**params, "economy.capital_output_ratio": k})[1], i_target, 1.0, 7.0, True,
            0.005, 12, log, "K/Y")
        u, inv = moments(params)
        if abs(u - u_target) <= 0.004 and abs(inv - i_target) <= 0.006:
            ok_u = ok_i = True
            break
    u, inv = moments(params)
    return CalibrationResult(
        iso3, u_target, params["government.income_tax_rate"], u, bool(ok_u and (ok_i or i_target is None)),
        i_target, params["economy.capital_output_ratio"] if i_target is not None else None,
        inv, [tuple(round(x, 5) if isinstance(x, float) else x for x in row) for row in log])


calibrate_income_tax = calibrate_country  # backwards-compatible name


def save_calibration(result: CalibrationResult) -> str:
    path = sources.REAL_DATA.parent / "calibration" / f"{result.iso3}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2))
    return str(path)


def load_calibration(iso3: str) -> dict | None:
    path = sources.REAL_DATA.parent / "calibration" / f"{iso3}.json"
    return json.loads(path.read_text()) if path.exists() else None
