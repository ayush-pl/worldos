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
                       "bank & central-bank balance sheets, goods & capital conservation, C+I+G = revenue, wages paid = received"), \
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
    qi = q.groupby("quarter").agg(i=("investment", "sum"), c=("consumption", "sum"), p=("cpi", "mean"))
    sd_i = float(np.log(qi.i / qi.p).diff(4).std())
    sd_c = float(np.log(qi.c / qi.p).diff(4).std())
    return [
        CheckResult("investment more volatile than consumption", "stylised fact", sd_i > sd_c,
                    f"sd(yoy real I) = {sd_i:.4f}, sd(yoy real C) = {sd_c:.4f}", "sd(I) > sd(C) (business-cycle fact)"),
        CheckResult("Okun's law", "stylised fact", okun < 0, f"corr(du, dlog Y) = {okun:+.3f}", "negative"),
        CheckResult("wage Phillips curve, single-run correlation (information only)", "information", True,
                    f"corr(u, wage growth yoy) = {phillips:+.3f}",
                    "not a pass/fail test: with anchored expectations the curve is flat and the in-sample "
                    "correlation is unstable across seeds; see the causal test below"),
    ]


def check_monetary_transmission(seeds=(1, 2, 3)) -> CheckResult:
    """Raise the policy rate by 3pp for two years after warm-up (vs baseline): investment should fall."""
    from worldos.scenario.experiment import ExperimentSpec, Intervention, ScenarioSpec, run_experiment

    spec = ExperimentSpec(
        name="rate_shock", seeds=list(seeds), warmup_months=36, horizon_months=24, summary_window_months=12,
        base_overrides={"population.n_person_records": 8000, "economy.n_firms": 100,
                        "central_bank.max_step": 0.0},
        variants=[ScenarioSpec(name="tight", interventions=[
            Intervention(at_month=0, path="central_bank.policy_rate", value=0.05)])],
        baseline=ScenarioSpec(name="baseline", interventions=[
            Intervention(at_month=0, path="central_bank.policy_rate", value=0.02)]),
        outcomes=["investment", "gdp_real", "unemployment_rate"])
    comp = run_experiment(spec).comparisons.set_index("outcome")
    d = comp.loc["investment"]
    return CheckResult("monetary transmission: +3pp policy rate lowers investment", "stylised fact",
                       bool(d.ci95_high < 0), f"investment diff {d.diff_mean:+.4g} [{d.ci95_low:+.3g}, {d.ci95_high:+.3g}]",
                       "95% CI below zero (rule frozen; fixed rates 5% vs 2%)")


def check_phillips_causal(seeds=(1, 2, 3)) -> CheckResult:
    """Demand shock (public spending 15% -> 18% of GDP): unemployment should fall while wages and inflation rise."""
    from worldos.scenario.experiment import ExperimentSpec, ScenarioSpec, run_experiment

    spec = ExperimentSpec(name="demand_shock", seeds=list(seeds), warmup_months=24, horizon_months=36,
                          variants=[ScenarioSpec(name="more_g", overrides={"government.spending_share": 0.18})],
                          outcomes=["unemployment_rate", "mean_wage", "inflation_yoy"])
    c = run_experiment(spec).comparisons.set_index("outcome")
    ok = c.loc["unemployment_rate", "ci95_high"] < 0 and c.loc["mean_wage", "ci95_low"] > 0 \
        and c.loc["inflation_yoy", "ci95_low"] > 0
    return CheckResult("wage Phillips curve, causal (demand shock)", "stylised fact", bool(ok),
                       f"du {c.loc['unemployment_rate', 'diff_mean']:+.4f}, dwage {c.loc['mean_wage', 'diff_mean']:+.2f}, "
                       f"dinflation {c.loc['inflation_yoy', 'diff_mean']:+.4f}",
                       "lower unemployment with higher wages and inflation (all 95% CIs exclude 0)")


def run_suite(seed: int = 1, months: int = 180) -> list[CheckResult]:
    acc, df = check_accounting(months, seed)
    results = [acc, check_reproducibility(seed), check_synthesis(seed), *check_micro_macro_demography(seed)]
    if acc.passed:
        results += check_stylised_facts(df)
        results.append(check_phillips_causal())
        results.append(check_monetary_transmission())
    return results


def format_results(results: list[CheckResult]) -> str:
    lines = ["| check | kind | result | value | criterion |", "|---|---|---|---|---|"]
    for r in results:
        status = "INFO" if r.kind == "information" else ("PASS" if r.passed else "FAIL")
        lines.append(f"| {r.name} | {r.kind} | {status} | {r.value} | {r.criterion} |")
    return "\n".join(lines)
