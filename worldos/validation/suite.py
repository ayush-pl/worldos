"""Alpha validation suite (``worldos validate``).

Three kinds of evidence, reported separately and never conflated:
* accounting    - identities that must hold exactly (to floating-point tolerance);
* consistency   - agreement between representation levels / reproducibility;
* stylised fact - qualitative macro regularities (Okun's law, wage Phillips curve).
                  Passing them is weak evidence of plausibility, NOT empirical validation.
Empirical validation against real data (historical reconstruction, out-of-sample
tests) requires the data pipeline milestones in docs/09 and is not claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldos.build import build_world, default_engines
from worldos.config import WorldConfig, apply_overrides
from worldos.core.simulation import Simulation
from worldos.population.demography import CohortProjection, DemographyEngine, aggregate_age_sex


@dataclass
class CheckResult:
    name: str
    kind: str
    passed: bool
    value: str
    criterion: str


def check_accounting(months: int, seed: int) -> tuple[CheckResult, pd.DataFrame]:
    world = build_world(WorldConfig(), seed)
    sim = Simulation(world, default_engines())
    try:
        sim.run(months)
        ok, msg = True, f"{months} ticks, 0 violations"
    except AssertionError as exc:  # InvariantViolation
        ok, msg = False, str(exc)[:200]
    return CheckResult("accounting invariants every tick", "accounting", ok, msg,
                       "bank & central-bank balance sheets, goods conservation, C+G = revenue, wages paid = received"), \
        sim.metrics_frame()


def check_reproducibility(seed: int, months: int = 24) -> CheckResult:
    a = Simulation(build_world(WorldConfig(), seed)).run(months).metrics_frame()
    b = Simulation(build_world(WorldConfig(), seed)).run(months).metrics_frame()
    same = a.drop(columns=["date"]).equals(b.drop(columns=["date"]))
    return CheckResult("bitwise reproducibility", "consistency", same, "identical" if same else "different",
                       "same config + seed => identical metric history")


def check_micro_macro_demography(seed: int, years: int = 20, records: int = 40_000) -> list[CheckResult]:
    cfg = apply_overrides(WorldConfig(), {"population.n_person_records": records})
    world = build_world(cfg, seed)
    cohort = CohortProjection.from_persons(world.persons, world.mortality, world.fertility)
    engine = DemographyEngine()
    for _ in range(years * 12):
        world.begin_tick()
        engine.step(world)
        world.clock.advance()
        cohort.step()
    f, m = aggregate_age_sex(world.persons, world.mortality.n_ages)
    micro_total = f.sum() + m.sum()
    rel = micro_total / cohort.total - 1
    groups = np.arange(world.mortality.n_ages) // (15 * 12)
    micro_g = np.bincount(groups, weights=f + m) / micro_total
    macro_g = np.bincount(groups, weights=cohort.female + cohort.male) / cohort.total
    tvd = 0.5 * np.abs(micro_g - macro_g).sum()
    return [
        CheckResult(f"micro vs cohort population total after {years}y", "consistency", abs(rel) < 0.02,
                    f"relative difference {rel:+.4f}", "|difference| < 2% (Monte Carlo noise only)"),
        CheckResult(f"micro vs cohort age structure after {years}y", "consistency", tvd < 0.02,
                    f"total variation distance {tvd:.4f}", "TVD over 15-year age groups < 0.02"),
    ]


def check_synthesis(seed: int) -> CheckResult:
    world = build_world(WorldConfig(), seed)
    from worldos.population.schedules import stable_population

    target = stable_population(world.mortality, world.fertility).total
    f, m = aggregate_age_sex(world.persons, world.mortality.n_ages)
    groups = np.arange(world.mortality.n_ages) // (5 * 12)
    got = np.bincount(groups, weights=f + m)
    got /= got.sum()
    want = np.bincount(groups, weights=target)
    tvd = 0.5 * np.abs(got - want).sum()
    return CheckResult("synthetic population vs target age distribution", "consistency", tvd < 0.03,
                       f"TVD {tvd:.4f}", "TVD over 5-year age groups < 0.03")


def check_stylised_facts(df: pd.DataFrame) -> list[CheckResult]:
    q = df.iloc[24:].copy()  # drop spin-up
    q["quarter"] = q["tick"] // 3
    qq = q.groupby("quarter").agg(u=("unemployment_rate", "mean"), y=("gdp_real", "sum"), w=("mean_wage", "mean"))
    okun = np.corrcoef(qq["u"].diff().iloc[1:], np.log(qq["y"]).diff().iloc[1:])[0, 1]
    annual_wage_growth = np.log(qq["w"]).diff(4)
    valid = annual_wage_growth.notna()
    phillips = np.corrcoef(qq["u"][valid], annual_wage_growth[valid])[0, 1]
    return [
        CheckResult("Okun's law", "stylised fact", okun < 0, f"corr(du, dlog Y) = {okun:+.3f}", "negative"),
        CheckResult("wage Phillips curve", "stylised fact", phillips < 0,
                    f"corr(u, wage growth yoy) = {phillips:+.3f}", "negative"),
    ]


def run_suite(seed: int = 1, months: int = 180) -> list[CheckResult]:
    acc, df = check_accounting(months, seed)
    results = [acc, check_reproducibility(seed), check_synthesis(seed), *check_micro_macro_demography(seed)]
    if acc.passed:
        results += check_stylised_facts(df)
    return results


def format_results(results: list[CheckResult]) -> str:
    lines = ["| check | kind | result | value | criterion |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {r.kind} | {'PASS' if r.passed else 'FAIL'} | {r.value} | {r.criterion} |")
    return "\n".join(lines)
