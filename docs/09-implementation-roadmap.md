# 09 · Implementation Roadmap (Task 9)

Milestones are planning targets, not promises; scientific validation and data availability may
change them. Each milestone has acceptance tests that must pass before the next builds on it.

## Phase 1 — Foundation (months 0–6) → WORLD-OS Alpha

| M | Deliverable | Status | Acceptance tests |
|---|---|---|---|
| M0 | Kernel: clock, scheduler, RNG streams, columnar tables, action kernel, provenance | ✅ | `tests/test_core.py` |
| M1 | Synthetic one-country world: geography, population, SFC economy, institutions, scenario lab, validation suite, CLI, HTML dashboard | ✅ | full pytest suite; `worldos validate` all PASS; 10⁶ records < 1 s/month |
| M2 | Data connectors: UN WPP, World Bank WDI, Natural Earth, climate records; checksummed extracts | ✅ (persistence in Parquet/PostGIS still ⏳) | checksummed raw files; provenance complete on 100% of observations; licence gate tested |
| M3 | Real-country population from WPP (empirical schedules, residual migration); hindcast and projection benchmarks | ✅ six countries (household fitting to census marginals still ⏳) | micro vs WPP age-sex structure TVD < 0.02 at base year; 10-year projection within WPP 80% interval |
| M4 | Calibration v1: moment matching / SBI of economic parameters to national accounts and labour statistics | ◐ one-parameter calibration to unemployment + untargeted-moment reports | target moments inside tolerance; posterior predictive checks; parameter identifiability report |
| M5 | API + run store + run manifests; minimal web dashboard | ⏳ | reproduce any stored run bit-for-bit from its manifest |

## Phase 2 — Economic civilisation (6–12)

Multi-sector input-output production; capital investment and depreciation; housing and mortgages;
consumer credit; multiple banks with an interbank market; firm entry/exit by sector; progressive
tax-benefit system from OECD parameters; labour market with regional commuting.
*Acceptance:* SFC invariants extended (capital, housing); stylised-facts battery (firm-size
distribution, investment volatility > consumption volatility, procyclical credit).

## Phase 3 — Environmental integration (12–24)

*Started:* climate emulator validated against observations; territorial CO2 accounting in country economies (docs/11).

Emissions accounting per sector; simple climate model coupling (FaIR-type) for global temperature;
energy engine (generation mix, dispatch, prices); agriculture with yield emulators; water balance;
land use/forest cover; damage channels to productivity and health.
*Acceptance:* energy and carbon conservation invariants; reproduction of historical emissions vs
Global Carbon Budget within stated tolerance; climate module matches its reference implementation.

## Phase 4 — Global civilisation (24–36)

Multiple countries; trade engine (BACI/EXIOBASE-calibrated); exchange rates; migration; international
institutions; distributed execution with geographic partitioning (Ray) and adaptive resolution.
*Acceptance:* world totals match sums of countries; trade balances sum to zero; distributed run equals
single-process run for the same seed.

## Phase 5 — World intelligence (36–48)

Historical reconstruction (e.g. 2000–2019) and out-of-sample validation; shock reconstruction
(2008, 2020); global sensitivity analysis; multi-objective policy optimisation with explicit
constraints; research agents automating experiment design.
*Acceptance:* published validation reports per engine and for the coupled model, with failures
documented.

## Phase 6 — Planetary scale (48–60+)

Global coverage at adaptive resolution; continuous data integration; higher-fidelity physical
coupling via external products; billion-person representation using L2/L3 levels.

## Cross-cutting tracks (every phase)

Testing (unit, property, statistical, regression on golden runs) · benchmarking (records/s per
engine) · security & governance (RBAC, audit, ethics review) · documentation of assumptions.

## Immediate next tasks

1. **Capital and investment** (closes the consumption-share gap and gives monetary policy a channel).
2. **Multi-moment calibration** (simulation-based inference over ~10 parameters; targets incl. Gini, inflation).
3. **Global sensitivity analysis** (Morris screening) of real-country worlds.
4. **Gridded population** (WorldPop/GHSL) for spatial allocation; **energy engine** behind emissions.
