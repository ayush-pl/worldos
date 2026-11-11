# 05 · Population and Agent Representation Strategy (Task 5)

Includes the **Agent Behaviour Specification** and **Agent Communication Protocol** (summaries).

## 1. Objectives

Represent billions of people credibly *without* billions of independent reasoning processes, while
keeping micro-level heterogeneity where it matters for the question being asked.

## 2. Three representation levels

| Level | Unit | State | Dynamics | Use |
|---|---|---|---|---|
| **L1 explicit** | one person / household (weight = 1) | full record | stochastic microsimulation | regions of interest, validation, small countries |
| **L2 weighted** | one record = *w* identical units | full record | same rules; flows scale with *w* | national simulation (Alpha default: w = 50) |
| **L3 aggregate** | cohort cell (sex × age-month × [region, education, status]) | expected counts | deterministic projection with the *same* rates | global background, fast scenario screening |

**Consistency between levels (implemented):**
* L2 → L3: `aggregate_age_sex()` sums record weights into cohort cells.
* L3 → L2: `synthesize_population()` samples records from cohort-cell distributions.
* Dynamics: `CohortProjection` applies exactly the microsimulation's monthly event order and rates,
  so it equals the **expectation** of L2. The validation suite checks that after 20 years the micro
  total differs from the cohort total by < 2% and the age structure by TVD < 0.02 (observed:
  −0.09% and 0.004).
* L1 is L2 with `represented_population = n_person_records`.

**Known limitation of L2:** records with weight *w* move in blocks of *w* (all *w* represented
women give birth in the same month, or none). Unbiased, but it adds variance and removes
within-record heterogeneity. Mitigation (planned): **record splitting** in regions of interest
(a weight-*w* record → *k* records of weight *w/k* with perturbed attributes) and **merging** in
background regions; both preserve totals exactly.

## 3. Adaptive resolution (planned, Stage 3–4)

Each geographic partition carries a resolution level (L1/L2/L3) that can change between ticks.
Moving L3 → L2 synthesises records conditioned on cohort cells; L2 → L3 aggregates. Money and
goods held by records are re-assigned so that sector balance sheets are unchanged (conservation
test required for every transition).

## 4. Agent classes and decision mechanisms

| Class | Agents | Decision mechanism | Alpha |
|---|---|---|---|
| **A behavioural** | persons, households, workers, ordinary firms | rules + statistical models; vectorised; later fitted to microdata or learned | ✅ consumption, job search, pricing, hiring, production, credit demand |
| **B strategic** | government, central bank, banks, large firms | transparent rules → optimisation/RL → (optional) LLM, always via the action kernel | ✅ Taylor-rule central bank, debt-rule fiscal policy |
| **C research** | analysts that design and run experiments | scripted experiment specs now; automated hypothesis search later | ◐ experiment runner |
| **D coordination** | scheduler, market clearing, consistency | kernel services | ✅ clock, scheduler, kernel, invariants |

## 5. Agent Behaviour Specification (Alpha rules)

All parameters are in `worldos/config.py` with provenance kind *assumed*.

**Households.** Expected disposable income Yᵉ ← (1−ω)Yᵉ + ω·Y (ω = 0.25). Consumption budget
C = c_y·Yᵉ + c_w·D, capped by deposits (c_y = 0.9, c_w = 0.02/month), i.e. the SFC consumption
function. Search: each round, sample *k* = 3 firms with stock and buy from the cheapest;
proportional rationing; 3 rounds.

**Persons (life course).** School until the leaving age of their education level (16/18/22);
then join the labour force with probability 0.8 or stay inactive; retire at 65; education is
inherited through a mobility matrix; leave the parental home at 20–34 at 10%/year.

**Unemployed workers.** Each round sample *k* firms with vacancies, apply to the highest wage
offer; firms accept applicants in random order until vacancies are filled.

**Firms.** Demand signals relative to the target inventory buffer: *short* (stock < ½ target) or
*glut* (stock > 1½ target). If short and price ≥ market average, raise expected demand; if
short and cheaper, raise the markup; if glut and cheaper, cut expected demand; if glut and dearer,
cut the markup (CATS logic, Assenza et al. 2015, applied to a cost-plus markup as in Dosi et al.
2010). Price = (1 + markup)·wage/productivity. Wage offer: indexation to positive past inflation
(50%) + trend productivity growth, plus a random rise when vacancies were unfilled (Lengnick 2013)
and a smaller random cut after layoffs (downward nominal rigidity). Labour demand = (target output)
/ productivity; fire excess > 5%; borrow to cover the planned wage bill; dividends only from
after-tax profit above liquidity and net-worth buffers. Bankrupt when illiquid or net worth < 0;
replaced after 3 months by an entrant financed by a bank start-up loan.

**Bank.** Lends up to equity / 8% (capital requirement); loan rate = policy + 2% + 3%·leverage;
deposit rate = max(0, policy − 1%); working-capital credit lines for payroll shortfalls; pays out
10% per month of capital above the 12% target.

**Central bank (class B).** Quarterly Taylor-type rule on inflation and unemployment gaps, gradual
(max 75 bp per decision), zero lower bound; validated against legal bounds.

**Government (class B).** Annual debt rule: Δτ = clip(0.05·(debt/GDP − 0.55), ±2 pp); public
consumption = 13% of trailing GDP; unemployment benefit 50% and pension 40% of mean wage.

## 6. Agent execution cycle (as implemented for class B)

1. *Observe* published indicators (one-tick lag) → 2. *update beliefs/memory* (policy object
state, cloned with the world) → 3. *evaluate* rule/optimiser → 4. *propose* `Action` with rationale
and `DecisionSource` → 5. kernel *validates* (bounds, rate-of-change, allowed actor) → 6. *apply*
via service or reject → 7. *audit* both cases → 8. consequences appear in next published metrics →
9. *learning* hook (not used in Alpha).

## 7. Agent Communication Protocol (summary)

* **Typed messages** (`MessageType`): product order, employment offer/contract, loan
  application/decision, investment offer, negotiation, trade agreement, policy announcement,
  institutional decision. Each has sender, recipient (or broadcast), tick and a payload schema.
* **High-volume interactions** (millions of orders per tick) are exchanged as **columnar order
  batches** inside market engines rather than per-message objects — same semantics, vectorised.
* **Actions vs messages:** a *message* informs; an *action* requests a state change and must pass
  the kernel. Accepted policy actions are broadcast as policy announcements.
* **Distributed setting:** cross-partition messages are buffered and delivered at the tick
  barrier, in a deterministic order (sorted by sender id, then sequence).

## 8. Privacy

Synthetic records are sampled from aggregate distributions; no record is derived from an
identifiable person. When microdata are used for calibration, only fitted parameters or
disclosure-controlled aggregates leave the secure environment.

## 9. Acceptance criteria

* L2↔L3 consistency tested (✅). Reproducibility tested (✅). Strategic decisions audited and
  validated; invalid LLM-sourced proposals rejected (✅ test).
