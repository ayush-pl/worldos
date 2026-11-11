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


def _cmd_data(args: argparse.Namespace) -> int:
    from worldos.countries.sources import check_integrity

    if args.action == "verify":
        bad = check_integrity()
        for src, files in bad.items():
            print(f"{src:<14} {'OK' if not files else 'CHANGED: ' + ', '.join(files)}")
        return 1 if any(bad.values()) else 0
    from worldos.data.connectors.climate import fetch_climate
    from worldos.data.connectors.naturalearth import fetch_naturalearth
    from worldos.data.connectors.wdi import fetch_wdi
    from worldos.data.connectors.wpp import fetch_wpp

    from worldos.data.connectors.kontur import fetch_kontur

    fetchers = {"wpp": fetch_wpp, "wdi": fetch_wdi, "naturalearth": fetch_naturalearth, "kontur": fetch_kontur}
    countries = args.countries.split(",") if args.countries else None
    for name in args.sources.split(","):
        if name == "climate":
            print("climate ->", fetch_climate(countries))
        else:
            print(name, "->", fetchers[name](countries))
    return 0


def _cmd_country(args: argparse.Namespace) -> int:
    from worldos.countries.sources import available_countries

    if args.action == "list":
        print(" ".join(available_countries()))
        return 0
    iso = args.iso3.upper()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == "calibrate":
        from worldos.countries.calibrate import calibrate_income_tax, save_calibration

        r = calibrate_income_tax(iso)
        print(f"{iso}: income tax {r.income_tax_rate:.4f}, unemployment target {r.target_unemployment:.4f}, "
              f"achieved {r.achieved_unemployment:.4f}, converged={r.converged} -> {save_calibration(r)}")
        return 0 if r.converged else 1
    if args.action == "validate":
        from worldos.countries.validate import country_report

        md = country_report(iso)
        (out / f"{iso}.md").write_text(md)
        print(md)
        return 0
    from worldos.core.simulation import Simulation
    from worldos.countries.build import build_country_world
    from worldos.report.html import render_dashboard

    world = build_country_world(iso, args.base_year, args.records, args.firms, args.seed)
    sim = Simulation(world).run(args.months)
    df = sim.metrics_frame()
    df.to_csv(out / "metrics.csv", index=False)
    from worldos.report.maps import population_map_svg, regional_summary

    regions = regional_summary(world)
    regions.to_csv(out / "regions.csv", index=False)
    table = regions.head(15).to_html(index=False, float_format=lambda v: f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}")
    extra = (f"<h2>Population map</h2>{population_map_svg(world)}"
             f"<h2>Largest regions at end of run</h2><div class='tablewrap'>{table}</div>")
    (out / "dashboard.html").write_text(render_dashboard(
        df, f"WORLD-OS: {world.country.name}",
        f"Real demographic/geographic data, calibrated simplified economy · seed {args.seed} · from {args.base_year}",
        extra_html=extra))
    print(f"{world.country.name}: simulated {args.months} months; outputs in {out}/ "
          "(metrics.csv, regions.csv, dashboard.html)")
    return 0


def _cmd_climate(args: argparse.Namespace) -> int:
    from worldos.climate.history import calibrate, climate_report, scenario

    if args.action == "validate":
        md, _ = climate_report()
        print(md)
        return 0
    r = calibrate()
    s = scenario(net_zero_year=args.net_zero, p=r["params"])
    print(s.loc[[2030, 2050, 2075, 2100]].round(2).to_string())
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
    d = sub.add_parser("data", help="fetch real data extracts or verify their checksums")
    d.add_argument("action", choices=["fetch", "verify"])
    d.add_argument("--sources", default="wpp,wdi,naturalearth,kontur,climate")
    d.add_argument("--countries", default="", help="comma-separated ISO3 codes (default: the six committed)")
    d.set_defaults(func=_cmd_data)
    k = sub.add_parser("country", help="real-country worlds: list | calibrate | validate | run")
    k.add_argument("action", choices=["list", "calibrate", "validate", "run"])
    k.add_argument("iso3", nargs="?", default="")
    k.add_argument("--base-year", type=int, default=2023)
    k.add_argument("--months", type=int, default=120)
    k.add_argument("--records", type=int, default=40_000)
    k.add_argument("--firms", type=int, default=400)
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--out", default="results/country")
    k.set_defaults(func=_cmd_country)
    m = sub.add_parser("climate", help="climate module: validate | scenario")
    m.add_argument("action", choices=["validate", "scenario"])
    m.add_argument("--net-zero", type=int, default=None, help="year global CO2 reaches zero (omit: constant)")
    m.set_defaults(func=_cmd_climate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
