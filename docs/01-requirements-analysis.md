# 01 · Requirements Analysis (Task 1)

**Status:** v0.1 draft · **Owner:** architecture · **Applies to:** whole programme

## 1. Objective

Turn the WORLD-OS vision into requirements that can be engineered, tested and prioritised, and
separate what is **engineering** (known methods, effort-bound) from what is **research** (open
scientific questions, outcome-uncertain).

## 2. Reading of the vision

WORLD-OS is three things at once, and each has different success criteria:

| Facet | What it means | Success looks like |
|---|---|---|
| **Coupled simulator** | Physical, ecological, demographic, economic and institutional sub-models exchanging state on a shared clock and geography | Conservation laws and accounting identities hold across couplings; sub-models are individually validated |
| **Data twin** | A curated, provenance-tracked estimate of the present and past world | Every value traceable to source, licence, resolution and uncertainty; observation/simulation separated |
| **Experimentation platform** | Branchable worlds, scenarios, policy experiments, uncertainty quantification | Reproducible experiments, ensembles, paired comparisons, honest reporting |

## 3. Functional requirements (condensed, traceable to spec sections)

| ID | Requirement | Spec § | Alpha |
|---|---|---|---|
| FR-GEO-1 | Hierarchical places (planet → cell) with stable ids, geometry, parent links | 3 | ✅ synthetic country, H3 |
| FR-GEO-2 | Change of spatial resolution (H3 parent/child) | 3 | ✅ |
| FR-POP-1 | Person/household records with demographic, economic and social state | 4 | ✅ subset (see 05) |
| FR-POP-2 | Life-cycle events: birth, ageing, schooling, labour-market entry, retirement, death, household formation/dissolution | 4 | ✅ (no partnership formation, no migration) |
| FR-POP-3 | Three representation levels with consistent transitions | 4 | ✅ weighted micro ↔ cohort aggregate; explicit (weight 1) by config |
| FR-ECO-1 | Households, firms, bank, government, central bank with stock-flow consistent accounting | 5, 6 | ✅ |
| FR-ECO-2 | Labour, goods and credit markets with decentralised matching | 5 | ✅ |
| FR-ECO-3 | National-accounts indicators: GDP, inflation, unemployment, distribution, poverty, debt | 5 | ✅ |
| FR-ECO-4 | Multi-sector production, capital investment, housing, international trade, exchange rates | 5 | ⏳ Phase 2/4 |
| FR-INS-1 | Strategic agents (government, central bank) whose decisions are validated before applying | 6, 10 | ✅ |
| FR-ENV-1 | Climate, water, agriculture, energy, biodiversity models coupled to the economy | 7 | ⏳ Phase 3 |
| FR-AGT-1 | Typed agent messages and actions; no direct writes from AI decisions | 10 | ✅ kernel + audit |
| FR-SIM-1 | Central clock, multi-rate scheduling, event scheduler, consistency checks | 11 | ✅ |
| FR-DAT-1 | Provenance on every value; observed / estimated / inferred / simulated / forecast (+ assumed) | 12 | ✅ store + ingestion |
| FR-DAT-2 | Ingestion pipelines for catalogue sources | 12 | ⏳ M2 (CSV ingestion + licence gate done) |
| FR-SCN-1 | Clone baseline, change assumptions, multi-seed ensembles, compare with uncertainty, report | 13 | ✅ |
| FR-POL-1 | Multi-objective policy optimisation with constraints | 14 | ⏳ Phase 5 |
| FR-VAL-1 | Accounting checks, cross-level consistency, sensitivity, UQ, historical reconstruction | 17 | ✅ first three partially; ⏳ rest |
| FR-UI-1 | Command centre with globe, explorers, labs | 18 | ⏳ (Alpha: static HTML dashboard + CLI) |
| FR-GOV-1 | RBAC, audit, reproducibility, model versioning, isolation from real systems | 19 | ◐ audit, reproducibility, isolation by design |

## 4. Non-functional requirements

* **Reproducibility:** identical config + seed + code version ⇒ bit-identical results (tested).
* **Scalability path:** columnar agent state; vectorised rules; partitionable by geography; target
  10⁶ records/core-month in Stage 1 (Alpha: ≈0.6 s per month for 10⁶ person records, 10³ firms).
* **Correctness:** accounting invariants checked every tick; strict mode aborts on violation.
* **Transparency:** every parameter documented with provenance kind; every policy action audited
  with its rationale and decision source.
* **Modularity:** engines are stateless transition functions over a single world state; adding an
  engine cannot perturb another engine's random numbers (named RNG streams).

## 5. Scientific requirements and where research is needed

| Area | Established foundations | Open research (cannot be promised by engineering alone) |
|---|---|---|
| Demography | Cohort-component projection, parametric mortality/fertility models, microsimulation | Endogenous fertility/migration behaviour under economic and climate stress |
| Macro-ABM | CATS, EURACE, K+S, SFC accounting (Godley & Lavoie) | Empirical validation of ABMs; calibration of high-dimensional models (e.g. surrogate/simulation-based inference) |
| Climate | Earth-system models (CMIP), emulators (FaIR, MAGICC) | Two-way coupling of agent behaviour with physical climate at fine resolution; damage functions |
| Agriculture/water | Crop models (e.g. DSSAT, APSIM), hydrological models | Robust global coupling with markets and land-use decisions |
| AI agents | RL, game theory, LLM agents | Validating LLM agent behaviour against human data; avoiding spurious "intelligence" |
| Integration | Co-simulation standards (e.g. FMI), coupling frameworks | Error propagation and uncertainty through coupled multi-scale models |

## 6. Computational requirements (order of magnitude)

* **8.2 × 10⁹ persons** at weight 1 would need ≈ 200 bytes/person of hot state ⇒ ~1.6 TB in memory,
  and ~10¹¹ rule evaluations per simulated month. That is feasible on a cluster for *simple*
  behavioural rules, but not with per-agent LLM calls (10¹⁰ calls/month is infeasible on cost and
  latency). Hence: weighted records + adaptive resolution, and LLMs only for a small number of
  strategic agents (docs/05).
* **Climate at km scale** (Destination Earth class) needs exascale HPC; WORLD-OS should *couple to*
  such products or emulators rather than re-run them.

## 7. Principal risks

1. **False precision** — a detailed-looking world implying predictive skill it lacks. *Mitigation:*
   provenance kinds, mandatory uncertainty, validation tiers, report caveats (implemented).
2. **Unvalidatable complexity** — too many interacting parameters to identify. *Mitigation:* validate
   each engine alone first; sensitivity analysis; prefer established sub-models.
3. **Data licensing** — some key sources are non-commercial or share-alike. *Mitigation:* licence
   gate in ingestion (implemented), catalogue review.
4. **Ethical misuse** — treating simulated policy outcomes as authority. *Mitigation:* human review,
   isolation from real systems, no identifiable individuals (docs/03 §9).

## 8. Acceptance criteria for this document

* Each spec section maps to at least one requirement ID with a status (done).
* Research gaps are listed separately from engineering tasks (done).
