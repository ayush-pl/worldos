"""Scenario laboratory: branch a common baseline, change assumptions, compare with uncertainty.

Protocol (spec section 13):
 1. build the baseline world for seed s and run a warm-up (spin-up) period;
 2. clone the warmed-up state once per arm (baseline + variants);
 3. apply each arm's overrides at the branch date and schedule its dated
    interventions (all pass through the action kernel and are audited);
 4. reseed every arm identically (common random numbers) and run the horizon;
 5. repeat for many seeds (stochastic realisations);
 6. compare arms with paired differences and t-based confidence intervals.

Results are conditional model experiments on a synthetic economy. Never
interpret a single arm or seed as a prediction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from worldos.build import build_world, default_engines
from worldos.config import WorldConfig, apply_overrides, set_path
from worldos.core.clock import SimDate
from worldos.core.simulation import Simulation

DEFAULT_OUTCOMES = [
    "unemployment_rate", "inflation_yoy", "gdp_real", "gdp_nominal", "gov_debt_to_gdp", "poverty_rate",
    "gini_income", "gini_wealth", "mean_wage", "policy_rate", "income_tax_rate",
]


class Intervention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    at_month: int = Field(0, ge=0, description="months after the branch point")
    path: str  # e.g. "government.spending_share" or "config.economy.productivity_growth"
    value: Any


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""
    overrides: dict[str, Any] = Field(default_factory=dict)  # config paths applied at the branch point
    interventions: list[Intervention] = Field(default_factory=list)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    question: str = ""
    baseline: ScenarioSpec = Field(default_factory=lambda: ScenarioSpec(name="baseline"))
    variants: list[ScenarioSpec]
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    warmup_months: int = Field(24, ge=0)
    horizon_months: int = Field(60, ge=1)
    summary_window_months: int = Field(12, ge=1)
    outcomes: list[str] = Field(default_factory=lambda: list(DEFAULT_OUTCOMES))
    base_overrides: dict[str, Any] = Field(default_factory=dict)  # applied before warm-up to every arm


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    timeseries: pd.DataFrame  # columns: arm, seed, tick, date, <metrics>
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)


def _prepare_arm(world, arm: ScenarioSpec) -> None:
    for path, value in arm.overrides.items():
        set_path(world.config, path, value)
        # instruments initialised from config must follow config overrides at the branch point
        section, _, key = path.partition(".")
        if section == "government" and hasattr(world.government, key):
            setattr(world.government, key, value)
    for iv in arm.interventions:
        world.events.schedule(world.clock.tick + iv.at_month, "scenario_intervention",
                              {"path": iv.path, "value": iv.value})


def run_experiment(spec: ExperimentSpec, config: WorldConfig | None = None) -> ExperimentResult:
    config = apply_overrides(config or WorldConfig(), spec.base_overrides)
    arms = [spec.baseline, *spec.variants]
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    frames = []
    for seed in spec.seeds:
        world = build_world(config, seed)
        Simulation(world, default_engines()).run(spec.warmup_months)
        branch_tick = world.clock.tick
        for arm in arms:
            w = world.clone()
            _prepare_arm(w, arm)
            w.rng.reseed(10_000 + seed)  # common random numbers across arms
            Simulation(w, default_engines()).run(spec.horizon_months)
            df = pd.DataFrame(w.metrics[branch_tick:])
            df.insert(0, "seed", seed)
            df.insert(0, "arm", arm.name)
            frames.append(df)
    ts = pd.concat(frames, ignore_index=True)
    result = ExperimentResult(spec, ts)
    result.summary, result.comparisons = summarise(ts, spec, spec.baseline.name)
    return result


def summarise(ts: pd.DataFrame, spec: ExperimentSpec, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    last_tick = ts["tick"].max()
    window = ts[ts["tick"] > last_tick - spec.summary_window_months]
    per_seed = window.groupby(["arm", "seed"])[spec.outcomes].mean(numeric_only=True)
    rows = []
    for arm, grp in per_seed.groupby(level="arm"):
        for outcome in spec.outcomes:
            v = grp[outcome].dropna().to_numpy()
            rows.append({"arm": arm, "outcome": outcome, "mean": v.mean() if len(v) else np.nan,
                         "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                         "p05": np.quantile(v, 0.05) if len(v) else np.nan,
                         "p95": np.quantile(v, 0.95) if len(v) else np.nan, "n_seeds": len(v)})
    summary = pd.DataFrame(rows)
    comp = []
    base = per_seed.xs(baseline, level="arm")
    for arm, grp in per_seed.groupby(level="arm"):
        if arm == baseline:
            continue
        g = grp.droplevel("arm")
        for outcome in spec.outcomes:
            d = (g[outcome] - base[outcome]).dropna().to_numpy()
            n = len(d)
            mean = d.mean() if n else np.nan
            if n > 1:
                se = d.std(ddof=1) / np.sqrt(n)
                half = stats.t.ppf(0.975, n - 1) * se if se > 0 else 0.0
            else:
                half = np.nan
            comp.append({"arm": arm, "outcome": outcome, "diff_mean": mean, "ci95_low": mean - half,
                         "ci95_high": mean + half, "n_seeds": n,
                         "sign_robust": bool(n > 1 and (mean - half > 0 or mean + half < 0))})
    return summary, pd.DataFrame(comp)


def load_experiment(path: str) -> ExperimentSpec:
    import tomllib

    with open(path, "rb") as fh:
        return ExperimentSpec.model_validate(tomllib.load(fh))


def render_markdown(result: ExperimentResult) -> str:
    s = result.spec
    lines = [
        f"# Experiment: {s.name}",
        "",
        f"**Question:** {s.question or '(not stated)'}",
        "",
        "> WORLD-OS Alpha simulates a *synthetic* country with assumed parameters. These results are",
        "> conditional model experiments, not forecasts or statements about any real economy.",
        "",
        f"- Seeds (stochastic realisations): {len(s.seeds)} ({', '.join(map(str, s.seeds))})",
        f"- Warm-up: {s.warmup_months} months; horizon after branch: {s.horizon_months} months",
        f"- Outcomes summarised as the mean of the final {s.summary_window_months} months of each run",
        "- Arms share warm-up state and random-number streams (common random numbers)",
        "",
        "## Arms",
        "",
    ]
    for arm in [s.baseline, *s.variants]:
        lines.append(f"- **{arm.name}**: {arm.description or '-'}; overrides `{json.dumps(arm.overrides)}`; "
                     f"interventions `{json.dumps([iv.model_dump() for iv in arm.interventions])}`")
    lines += ["", "## Outcomes by arm (mean across seeds, [5th, 95th] percentile)", ""]
    pivot_rows = ["| outcome | " + " | ".join([s.baseline.name, *[v.name for v in s.variants]]) + " |",
                  "|---|" + "---|" * (1 + len(s.variants))]
    for outcome in s.outcomes:
        cells = []
        for arm in [s.baseline.name, *[v.name for v in s.variants]]:
            r = result.summary[(result.summary.arm == arm) & (result.summary.outcome == outcome)]
            if r.empty or pd.isna(r["mean"].iloc[0]):
                cells.append("n/a")
            else:
                r = r.iloc[0]
                cells.append(f"{r['mean']:.4g} [{r['p05']:.4g}, {r['p95']:.4g}]")
        pivot_rows.append(f"| {outcome} | " + " | ".join(cells) + " |")
    lines += pivot_rows
    lines += ["", "## Differences vs baseline (paired by seed, 95% t-interval)", "",
              "| arm | outcome | mean difference | 95% CI | sign robust across seeds? |", "|---|---|---|---|---|"]
    for _, r in result.comparisons.iterrows():
        lines.append(f"| {r['arm']} | {r['outcome']} | {r['diff_mean']:+.4g} | "
                     f"[{r['ci95_low']:+.4g}, {r['ci95_high']:+.4g}] | {'yes' if r['sign_robust'] else 'no'} |")
    lines += ["", "## Caveats", "",
              "- Intervals reflect only stochastic variation across seeds, not parameter or structural uncertainty.",
              "- Parameters are assumptions (provenance kind `assumed`); see docs/06 for the model and its limits.",
              "- Alpha has no capital investment, foreign sector, housing or environmental feedbacks.", ""]
    return "\n".join(lines)
