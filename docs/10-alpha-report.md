# 10 · WORLD-OS Alpha — Implementation Report (Task 10)

## What was built

A working, tested Python package that simulates one **synthetic** country month by month:
geography on H3, a weighted synthetic population with a demographic microsimulation and a cohort
twin, a stock-flow consistent economy (households, firms, one bank, a central bank, a government),
validated policy agents, a scenario laboratory with ensembles and confidence intervals, a validation
suite, a provenance-tracked data layer with a 30-dataset catalogue, a CLI and an HTML dashboard.

## Why each part was required

| Component | Why |
|---|---|
| Kernel (clock, engines, RNG streams, World.clone) | Every later engine plugs into it; branching worlds is the core of experimentation |
| Action kernel | The spec forbids AI decisions writing state directly; this is the enforcement point |
| Ledger + invariants | Economic results are meaningless if money or goods leak; checked every tick |
| Demography + cohort twin | Population is the base of every human system; the twin makes level transitions testable |
| SFC economy | Spec requires accounting identities as the foundation of economic calculations |
| Scenario lab | The main product: comparisons with uncertainty, never single forecasts |
| Data layer | Separation of observation and simulation, licensing discipline |

## Scientific foundations

Siler mortality; Hadwiger fertility; Lotka stable population; cohort-component projection;
Godley–Lavoie SFC accounting and consumption function; CATS (Assenza et al. 2015) adaptive
firm behaviour; Dosi et al. (2010) cost-plus pricing and wage norm; Lengnick (2013) wage
adaptation; Taylor (1993)-type rule; EU at-risk-of-poverty definition; rank-size rule for cities.
Details: docs/06.

## Results of the validation suite (`python -m worldos validate`, seed 1, 180 months)

| Check | Kind | Result |
|---|---|---|
| Accounting invariants every tick | accounting | PASS (0 violations) |
| Bitwise reproducibility | consistency | PASS |
| Synthetic population vs target age distribution | consistency | PASS (TVD 0.011) |
| Micro vs cohort population after 20 years | consistency | PASS (−0.09%) |
| Micro vs cohort age structure | consistency | PASS (TVD 0.004) |
| Okun's law | stylised fact | PASS (−0.99, near-mechanical: output ∝ employment) |
| Wage Phillips curve | stylised fact | PASS (−0.85) |

Performance (single core): 10⁵ records + 10³ firms ≈ 0.04 s/month; 10⁶ records + 10³ firms ≈ 0.6 s/month.

## Example experiment (examples/experiments/public_spending_increase.toml, 5 seeds)

Raising public consumption from 13% to 15% of trailing GDP lowered unemployment by 4.4 pp
(95% CI −5.7 to −3.1), raised CPI inflation by 1.2 pp (0.6 to 1.8), raised real output by about 5%,
and left the debt ratio statistically unchanged after five years (more tax revenue and higher
nominal GDP offset the extra spending). **This is a property of a synthetic model with assumed
parameters, not evidence about any real economy.**

## How it was debugged (worth recording)

The first calibrations collapsed or overheated. The fixes were all *model* corrections with a
literature basis, not curve-fitting: reserve remuneration and government interest (the bank was
structurally loss-making), working-capital credit lines, dividends restricted to after-tax profit
above buffers (firms had been paying dividends out of loans), demand signals relative to the target
inventory (firms with a buffer always looked "in glut"), cost-plus pricing (the profit share had
drifted upward), and downward nominal wage rigidity plus a productivity wage norm (deflation
spiral). The remaining fiscal tuning is documented as calibration (docs/06 §5).

## Tests

`python -m pytest` runs 33 tests: kernel, geo, demography (including a statistical micro/macro test),
economy (invariants, ledger correctness, injected-error detection, strict mode, kernel rejection of
an out-of-bounds LLM-sourced action), scenarios (reproducibility, clone independence, placebo arm =
zero difference, intervention auditing, rejected interventions, sign of a fiscal multiplier), data
(catalogue, licence gate, ingestion), statistics and the CLI.

## Known limitations

See docs/06 §8. In addition: Alpha is not calibrated to any real country, does not ingest real
data, and has no environmental, international, housing or capital dynamics. Its policy agents are
simple transparent rules; no learning and no LLMs are used. Observed model artefacts to fix:
the bank's equity/loans ratio drifts well above its 12% target in long runs (its payout rule is too
slow when interest income rises), and Okun's law holds near-mechanically because output is
proportional to employment.

## Next milestone

M2 → M3: real demographic data (UN WPP) for one real country, followed by calibration (M4).
