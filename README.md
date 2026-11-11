# WORLD-OS

**World Operating System: a planetary digital-twin and civilisation-simulation research programme.**
*Observe the Earth. Understand the World. Simulate the Future.*

This repository contains the **specification documents** for WORLD-OS (`docs/`) and **WORLD-OS
Alpha** (`worldos/`), a working and tested prototype that simulates **one synthetic country**.

> **Scope, stated plainly.** Alpha is *not* a digital twin of Earth. It is the foundation such a
> system needs: a reproducible simulation kernel, a demographic microsimulation, a stock-flow
> consistent agent-based economy with institutions, a validated action pathway for AI agents, a
> scenario laboratory with uncertainty, and a provenance-aware data layer. Its parameters are
> documented assumptions, and its outputs are model experiments, not forecasts.

## Quick start

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest
```

```bash
python -m worldos validate
```

```bash
python -m worldos run --months 120 --seed 1 --out results/run
```

```bash
python -m worldos experiment examples/experiments/public_spending_increase.toml --out results/g
```

```bash
python -m worldos catalogue
```

`run` writes `metrics.csv`, `audit_log.json` (every policy action, accepted or rejected),
`config.json` and a self-contained `dashboard.html`.

## What is implemented (Alpha)

| Area | Implemented | Not yet (planned, see docs/09) |
|---|---|---|
| Kernel | monthly clock, multi-rate engines, event queue, named RNG streams, world cloning, per-tick invariant checks | distributed execution |
| Geography | H3 hierarchy (planet → country → region → city → cells), resolution changes, synthetic territory | real boundaries, physical Earth layers |
| Population | Siler mortality, Hadwiger fertility, Lotka stable start, weighted microsimulation, cohort-component twin, schooling, labour entry, retirement, household formation & inheritance | migration, partnership formation, health |
| Economy | firms, households, bank, central bank, government; labour, goods and credit markets; SFC accounting; GDP (two sides), CPI, unemployment, Gini, poverty | multi-sector, capital, housing, trade, FX |
| Institutions | Taylor-rule central bank, debt-rule fiscal policy, validated through an action kernel with audit | optimisation/RL/LLM policies (interface ready) |
| Experiments | branch from warm-up, overrides & dated interventions, multi-seed ensembles with common random numbers, paired 95% CIs, markdown report | sensitivity analysis, optimisation |
| Data | 30-dataset catalogue with licence status, observation store with provenance, CSV ingestion with licence gate | live connectors (WPP, WDI next) |
| Environment | — | climate, energy, water, agriculture, biodiversity (Phase 3) |

Performance on one CPU core: 10⁶ person records and 10³ firms take about 0.6 s per simulated month,
with all accounting invariants checked every month.

## Documentation

Start with [docs/README.md](docs/README.md). Key documents: requirements (01), gap analysis (02),
architecture (03), ontology (04), agents (05), engines & accounting & validation (06), data
catalogue (07), MVP structure (08), roadmap (09), Alpha report (10).

## Principles

1. No identifiable real individuals: all populations are synthetic.
2. Agents propose and the kernel disposes. AI or LLM decisions never write state directly.
3. Every value carries its epistemic status (observed, estimated, inferred, simulated, forecast or assumed).
4. Accounting identities and conservation laws are checked, not assumed.
5. Futures are explored as ensembles of scenarios, never presented as predictions.
6. Simulated policies have no path to real-world systems.
