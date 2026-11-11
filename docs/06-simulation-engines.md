# 06 · Simulation Engines and Mathematical Foundations (Task 6)

Includes the **Economic Accounting Specification** and the **Scientific Validation Framework**
(summaries). Alpha engines are marked ✅; specified-only engines are marked ⏳.

## 1. Engine catalogue

| # | Engine (spec §11) | Method | Alpha |
|---|---|---|---|
| 1 | Geographic | H3 DGGS hierarchy, Voronoi regionalisation, rank-size settlements | ✅ synthetic |
| 4–5 | Population / demographic | Monthly stochastic microsimulation + cohort-component twin | ✅ |
| 6 | Human behaviour | Rule-based class-A behaviour (vectorised) | ✅ economic behaviours |
| 8 | Economic | Agent-based markets (labour, goods), cost-plus pricing with adaptive markup | ✅ |
| 9 | Financial | Single bank, endogenous money, capital requirement, central bank | ✅ |
| 10 | Government | Tax-benefit system, fiscal rule | ✅ |
| 19 | Technology | Exogenous labour-augmenting productivity growth | ✅ (exogenous only) |
| 21 | Scenario experimentation | Branch/clone, ensembles, paired comparisons | ✅ |
| 22 | AI decision & coordination | Action kernel, policies, audit | ✅ |
| 2–3 | Climate / weather | Couple to a simple climate model (FaIR-type) for global temperature; reanalysis/downscaled fields for regional weather | ⏳ Phase 3 |
| 7 | Social network | Household/kinship + workplace + spatial networks | ⏳ |
| 11 | International trade | Multi-country goods markets, gravity priors, exchange rates | ⏳ Phase 4 |
| 12–15 | Agriculture, energy, water, biodiversity | Crop-yield emulators, energy dispatch/LCOE, water balance, habitat indices | ⏳ Phase 3 |
| 16–18 | Healthcare, urban, transport | Health states, land/housing markets, network flows | ⏳ |
| 20 | Resource allocation | Optimisation (OR-Tools/Pyomo) | ⏳ Phase 5 |

## 2. Demography (✅)

**Mortality — Siler (1979):** μ(x) = a₁e^(−b₁x) + a₂ + a₃e^(b₃x), males × 1.25. Monthly death
probability q = 1 − exp(−μ(x + ½ month)/12); certain death at 110. Defaults give e₀ ≈ 77.1 (F) and
74.4 (M) years.

**Fertility — Hadwiger (1940) / Chandola, Coleman & Hiorns (1999):**
f(x) ∝ (R/x)^{3/2} exp(−H²(R/x + x/R − 2)) on [15, 50), numerically normalised so that
Σ f/12 = TFR exactly (tested). Default TFR 2.05, mean age at childbearing ≈ 28.5. Sex ratio at
birth 1.05.

**Initial population — Lotka stable population:** solve Σₐ e^(−r a) l_f(a) m_f(a) = 1 for r (Brent's
method), then c(a) ∝ e^(−r a) l(a). The synthetic pyramid is therefore *implied* by the stated
rates, not separately invented.

**Event order per month:** ageing → mortality → fertility → school/labour/retirement transitions →
household formation → estate settlement (deceased household's deposits and ownership pass to a
random living heir household, conserving money).

## 3. Economy (✅) — see docs/05 §5 for behavioural rules

**Production:** Yᵢ = Aᵢ·Lᵢ, Lᵢ = Σ_{workers} wⱼ·h(eduⱼ), h = (0.75, 1, 1.5). Aᵢ grows at g = 1.5%/yr
with N(0, 0.002) monthly firm noise.

**Prices:** pᵢ = (1 + μᵢ)·wᵢ/Aᵢ with μᵢ ∈ [0, 0.5] adapted by CATS signals.

**Credit:** granted up to min(financing need, 6 × monthly revenue − existing debt), scaled
pro-rata if aggregate demand for credit exceeds the bank's headroom E/κ − L (κ = 0.08).

## 4. Economic Accounting Specification (✅)

Balance-sheet matrix (single bank, closed economy, central bank with zero equity):

|  | Households | Firms | Bank | Government | Central bank | Σ |
|---|---|---|---|---|---|---|
| Deposits | +D_h | +D_f | −D | | | 0 |
| Loans | −L_h | −L_f | +L | | | 0 |
| Reserves (net) | | | +R | | −R | 0 |
| Government account (net) | | | | +G | −G | 0 |
| Inventories | | +pK | | | | +pK |
| Net worth | NW_h | NW_f | E | G | 0 | pK |

**Payment settlement:** private↔private moves deposits; government↔private moves deposits **and**
reserves. Interest: the central bank pays the policy rate on reserves and charges it on the
government overdraft; since R + G = 0 its income is zero.

**Invariants checked every tick:**
1. Bank: R + L − D = E_tracked, where E_tracked is updated only by the bank's income statement
   (loan interest + reserve interest − deposit interest − write-offs − dividends). A one-sided
   deposit change is detected (tested).
2. Central bank: R + G = 0.
3. Goods: closing stock = opening stock + produced − sold − written off.
4. Expenditure = revenue: C + G_purchases = Σ firm revenue; wages paid = wages received; hence
   GDP(expenditure) = C + G + ΔInv = GDP(income) = wages + operating surplus.
5. Feasibility: no negative deposits/inventories/loans, positive prices and wages, finite aggregates.
6. Population: employed ⇒ living employer; living person ⇒ living household; ownership shares sum to 1.

**National accounts definitions:** ΔInv valued at current unit cost; real GDP = units produced;
CPI = unit value of final sales; poverty = share of persons below 60% of median equivalised
(√size) disposable income.

## 5. Baseline calibration (Alpha)

Parameters are assumptions. The fiscal stance determines the stationary level of activity in SFC
models (output adjusts until taxes = government spending + transfers + interest, given private
saving targets). The tax rate, public consumption share and debt target were therefore tuned on a
small grid (3 seeds × 20 years) to stylised targets: unemployment ≈ 6–8%, inflation 1–2%, stable
debt ratio. The tuning reproduces the targets across seeds (see docs/10). This is **calibration to
stylised targets**, not estimation; replacing it with estimation against real data (simulation-based
inference) is milestone M4.

## 6. Scenario statistics (✅)

For outcome Y, arm a, seed s: Ȳ_{a,s} = mean over the final window. Arms share warm-up state and
random streams, so paired differences d_s = Ȳ_{a,s} − Ȳ_{base,s} remove common noise; report
mean(d) ± t_{0.975,n−1}·sd(d)/√n and flag "sign robust" when the interval excludes 0. Intervals cover
seed variability only — parameter and structural uncertainty need separate sensitivity analysis.

## 7. Scientific Validation Framework (summary)

| Tier | Question | Alpha status |
|---|---|---|
| V0 accounting | Do identities and conservation laws hold every tick? | ✅ enforced |
| V1 internal consistency | Do representation levels agree; is the model reproducible? | ✅ |
| V2 stylised facts | Okun's law, wage Phillips curve, firm-size skew, business-cycle co-movements | ◐ first two (Okun is near-mechanical in Alpha because output ∝ employment) |
| V3 calibration | Does the model match target moments of a real economy? | ⏳ M4 |
| V4 out-of-sample | Does a model calibrated to 1990–2010 reproduce 2010–2019? | ⏳ Phase 5 |
| V5 shock reconstruction | Does the integrated model reproduce 2008 or 2020 dynamics? | ⏳ Phase 5 |
| V6 benchmarking | Agreement with established models (UN WPP projections, FaIR, IAMs) | ⏳ |
| V7 sensitivity & UQ | Global sensitivity (Sobol/Morris), ensembles | ◐ seed ensembles only |
| V8 independent review | External scientific review | ⏳ |

## 8. Known limitations (Alpha)

(Updated: capital and investment, migration for real countries, and energy are now implemented — see
docs/11-12.) Single good; no housing, foreign sector or exchange rate; single bank; government spending purely consumption; labour market national (no commuting geography);
exogenous productivity; no environment. Monetary policy works only weakly (no interest-sensitive
investment or housing), so the economy can sit at the zero lower bound after demand shocks — a
property of this model, not a claim about real economies.
