# 12 · Capital and Investment, Two-Target Calibration, Gridded Population, Energy

**Status:** implemented. Reports: [`reports/countries/`](../reports/countries/), [`reports/energy.md`](../reports/energy.md).

## 1. Capital and investment (Phase 2)

**Production:** Y = min(A·L, κ·K), with κ = 1 / (12 · K/Y · u*). Defaults: capital-output ratio 3 years,
depreciation 6%/yr, target utilisation 0.85 (assumed; calibrated per country, see §2).

**Investment (flexible accelerator with finance):** desired capital K* = normal demand / (κ u*), where
"normal demand" is a slow-moving average of sales (smoothing 1/24 per month; using current
expectations produced a strong multiplier-accelerator cycle). Planned investment is
I = m · max(0, δK + φ(K* − K)) with φ = 1/36 per month and m = clip(1 + β(r_K − r_loan), 0, 2): the
return on capital relative to the loan rate scales investment (β = 4). Investment is financed after wages
and debt service, from retained earnings and bank loans (debt limit: max(6 months revenue, 50% of capital
value)), and bought from other firms through the goods market. Capital survives bankruptcy (taken over by
the entrant).

**Pricing:** full-cost: p = (1 + μ)(w/A + δ · K/Y · p̄); markups adapt by the CATS rule and revert towards
the normal markup at 3%/month (competition; without reversion markups ratcheted up in every boom).

**Monetary transmission:** via investment (β) and an intertemporal consumption channel (the propensity to
consume out of income falls by 1.0 × the real deposit rate gap; assumed). Wage expectations are anchored:
50% past inflation, 50% central-bank target.

**Fiscal rule:** debt-ratio gap outside a ±10 pp tolerance band, adjustment 0.01 per unit gap per year, max
1 pp per year; optional inflation response (off by default). The debt target is the initial ratio.

**Accounting:** GDP = C + I + G + ΔInventories; new invariant: capital(t) = capital(t−1) + investment − depreciation.

**What it fixed:** household consumption shares are now close to data without being targeted (USA 0.65 vs
0.68, Japan 0.53 vs 0.54, Brazil 0.65 vs 0.63, India 0.60 vs 0.57); labour share falls to ~0.5-0.6;
monetary policy has a measurable effect (validation: +3 pp policy rate lowers investment, 95% CI below zero).

**Baseline calibration (synthetic):** with capital the fiscal stance had to be recalibrated (tax 0.47,
public consumption 15%), giving u ≈ 6-7%, inflation ≈ 1.3% (sd 0.5%), investment ≈ 20% of GDP over 20 years.

**Debugging record (what failed and why):** fixed-expectation accelerator → 3-4 year boom-bust cycles;
debt-only fiscal rule + inflation erosion → tax-cut spiral; markup ratchet → stagflation; too small a
household wealth target → structural surpluses inconsistent with any debt target (sector balances).

**Validation change:** the single-run wage-Phillips correlation is no longer a pass/fail test. With anchored
expectations the curve is flat and the correlation varied from −0.25 to +0.35 across seeds; it is replaced by
a causal test (a demand shock lowers unemployment and raises wages and inflation; all CIs exclude zero) and
the correlation is reported as information.

## 2. Two-target calibration (M4)

Gauss-Seidel alternation of two bisections: income-tax rate → ILO unemployment; capital-output ratio → WDI
gross capital formation share. All six countries converge (Nigeria: tax only, no investment target in WDI).

| country | tax | K/Y | unemployment target / achieved | investment share target / achieved |
|---|---|---|---|---|
| India | 0.19 | 6.25 | 0.042 / 0.042 | 0.345 / 0.343 |
| USA | 0.35 | 3.81 | 0.036 / 0.037 | 0.216 / 0.215 |
| Germany | 0.53 | 4.00 | 0.031 / 0.033 | 0.220 / 0.221 |
| Japan | 0.56 | 5.12 | 0.026 / 0.026 | 0.278 / 0.276 |
| Brazil | 0.42 | 2.88 | 0.080 / 0.081 | 0.158 / 0.156 |
| Nigeria | 0.22 | – | 0.031 / 0.029 | – |

India's K/Y of 6.25 is implausibly high (real estimates are nearer 3-4): real economies reach a 34%
investment share through fast growth, which the model's demand-smoothing investment rule does not
anticipate, so K/Y absorbs it — and India's calibrated world overheats (≈12% inflation). Growth expectations
in investment are the next calibration improvement.

## 3. Gridded population (Kontur, H3)

Kontur Population 2023 (H3 resolution 8, CC BY 4.0) is aggregated to each country's model resolution
(936 KB committed). Households are placed on cells in proportion to population; cells within a ring of a major
city form that city. Totals agree with WPP within 1-3%; regional shares match known figures closely
(e.g. California ≈ 12%, Nordrhein-Westfalen ≈ 21%, Bayern ≈ 16%, São Paulo ≈ 22%). Country runs write
`regions.csv` and a population-density map in `dashboard.html`.

## 4. Energy model (Phase 3)

Kaya identity: primary energy = intensity × real output (intensity trend fitted on the last 10 years);
wind+solar share follows a logistic diffusion anchored at the observed base-year share; nuclear, hydro, bio
fixed; fossil fuels fill the rest in base proportions; optional coal phase-out year. CO2 = fossil energy ×
IPCC 2006 default factors × a reconciliation factor k (0.89-1.03 for the five countries with data: the
factors explain nearly all fossil CO2).

**Out-of-sample backtest** (fit 2008-2018, project 2019-2023 from observed GDP): primary energy within
1.6-6.1% (MAPE) and fossil CO2 within 3-5% for India, USA, Germany and Japan; Brazil's CO2 is off by 10.5%.
Wind+solar diffusion is overestimated for Japan and Brazil (logistic fits before the inflection point are
weakly identified). Nigeria lacks a source breakdown and keeps the intensity-based CO2 model.

**Example scenario** (`examples/experiments/india_energy_transition.toml`, 3 seeds, 5 years):
coal phase-out by 2040 → −168 Mt CO2/yr; plus twice-as-fast wind+solar diffusion → −446 Mt CO2/yr. There is
no energy-cost feedback to the economy yet, so GDP and unemployment are unchanged by construction.

**A bug the tests now guard:** doubling a logistic rate around a fixed midpoint *lowers* the curve before the
midpoint; diffusion is therefore anchored at today's observed share (test: faster rate ⇒ more renewables).

## 5. Next steps

1. Energy prices and costs feeding firm costs (makes energy policy affect the economy).
2. Growth expectations in investment (fixes India's K/Y) and simulation-based multi-moment calibration.
3. Climate damages to productivity; regional climate from the gridded layer.
4. Regional labour markets (commuting) using the gridded population.
