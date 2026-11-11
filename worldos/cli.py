"""Command-line interface: ``python -m worldos <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from worldos.build import build_world, default_engines
    from worldos.config import load_config
    from worldos.core.simulation import Simulation
    from worldos.report.html import render_dashboard

    cfg = load_config(args.config)
    t0 = time.time()
    world = build_world(cfg, args.seed)
    sim = Simulation(world, default_engines()).run(args.months)
    df = sim.metrics_frame()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics.csv", index=False)
    audit = [{**asdict(a), "source": a.source.value} for a in world.kernel.audit]
    (out / "audit_log.json").write_text(json.dumps(audit, indent=1, default=str))
    (out / "config.json").write_text(cfg.model_dump_json(indent=1))
    (out / "dashboard.html").write_text(render_dashboard(
        df, "WORLD-OS Alpha run", f"{cfg.geo.country_name} · seed {args.seed} · {args.months} months from {cfg.simulation.start}"))
    last = df.iloc[-1]
    print(f"Simulated {args.months} months ({len(world.persons)} person records, {int(last['firms_active'])} active firms) "
          f"in {time.time() - t0:.1f}s; invariant violations: {len(sim.violations)}")
    for key in ("unemployment_rate", "inflation_yoy", "gov_debt_to_gdp", "poverty_rate", "gini_income"):
        print(f"  {key:>22}: {last[key]:.4f}")
    print(f"Outputs written to {out}/ (metrics.csv, audit_log.json, config.json, dashboard.html)")
    return 0


def _cmd_experiment(args: argparse.Namespace) -> int:
    from worldos.scenario.experiment import load_experiment, render_markdown, run_experiment

    spec = load_experiment(args.spec)
    if args.seeds:
        spec.seeds = list(range(1, args.seeds + 1))
    result = run_experiment(spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.timeseries.to_csv(out / "timeseries.csv", index=False)
    result.summary.to_csv(out / "summary.csv", index=False)
    result.comparisons.to_csv(out / "comparisons.csv", index=False)
    report = render_markdown(result)
    (out / "report.md").write_text(report)
    print(report)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from worldos.validation.suite import format_results, run_suite

    results = run_suite(seed=args.seed, months=args.months)
    print(format_results(results))
    hard_fail = [r for r in results if not r.passed and r.kind in ("accounting", "consistency")]
    return 1 if hard_fail else 0


def _cmd_catalogue(args: argparse.Namespace) -> int:
    from worldos.data.catalogue import load_catalogue

    for e in load_catalogue():
        print(f"{e.id:<26} {e.license_status:<7} {e.license:<45} {e.url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worldos", description="WORLD-OS Alpha (synthetic-country prototype)")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run one simulation and write metrics, audit log and dashboard")
    r.add_argument("--months", type=int, default=120)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--config", default=None, help="TOML file mirroring WorldConfig sections")
    r.add_argument("--out", default="results/run")
    r.set_defaults(func=_cmd_run)
    e = sub.add_parser("experiment", help="run a multi-seed scenario experiment from a TOML spec")
    e.add_argument("spec")
    e.add_argument("--seeds", type=int, default=0, help="override: use seeds 1..N")
    e.add_argument("--out", default="results/experiment")
    e.set_defaults(func=_cmd_experiment)
    v = sub.add_parser("validate", help="run the Alpha validation suite")
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--months", type=int, default=180)
    v.set_defaults(func=_cmd_validate)
    c = sub.add_parser("catalogue", help="list the data catalogue")
    c.set_defaults(func=_cmd_catalogue)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
