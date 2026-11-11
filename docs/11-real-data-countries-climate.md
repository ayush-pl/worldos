# 11 · Real Data, Real-Country Worlds and the Climate Module

**Status:** implemented (milestones M2, M3, first step of M4, Phase 3 start). Reports:
[`reports/countries/`](../reports/countries/) and [`reports/climate.md`](../reports/climate.md).

## 1. Data pipeline (M2)

| Connector | Module | Method |
|---|---|---|
| UN WPP 2024 | `data/connectors/wpp.py` | streams 60-300 MB bulk files, keeps selected countries |
| World Bank WDI | `data/connectors/wdi.py` | API v2, 24 indicators, 1990-2024, retries with backoff |
| Natural Earth | `data/connectors/naturalearth.py` | admin-0/1 + populated places, simplified geometry |
| Climate records | `data/connectors/climate.py` | GCB emissions (OWID), NOAA CO2, HadCRUT5, AR6 forcing |

Every extract has a manifest with URL, time, filter and SHA-256; a test fails if a committed extract
changes. Licences and attributions: [`data/README.md`](../data/README.md).

## 2. Real-country demography (M3)

* **Initial population:** WPP 1 January single-age population by sex, spread uniformly over months.
* **Rates by calendar year:** WPP life-table m(x) (constant hazard within each year of age) and
  single-age ASFR. The ASFR file covers 15-49 only, so it is rescaled to the published TFR
  (births to mothers <15 and 50+ are redistributed over 15-49). Empirical schedules reproduce WPP
  life expectancy within 0.1 years and TFR exactly (tests).
* **Migration:** net migrants by sex and single age from the demographic residual method in
  estimation years (agrees with WPP's reported net migration to within 15% on decade averages; tested
  for the USA); Rogers-Castro age pattern scaled to WPP totals in projection years. Immigrant
  records form new households (adults) or join households (children); emigrants leave their assets
  in their household (assumption).
* **Mortality after 2023:** proportional shift of 2023 m(x) solved to match WPP projected life
  expectancy by sex.

**Hindcast 2000 → 2024 (same rates as WPP, so this tests the engine, not forecasting skill):**
cohort-model population error within ±0.3% for all six countries; weighted microsimulation within
±0.8% with 60,000 records (difference from the cohort model is Monte Carlo noise: mean −0.1%,
sd 0.3% across 8 seeds for Germany). Age-structure TVD ≤ 0.002.

**Projection 2024 → 2050 vs WPP medium variant:** within 1.3% for all six countries; the largest
gaps (USA, Germany, Nigeria ≈ −1%) come from the simplified mortality shift and the generic
Rogers-Castro migration age pattern.

## 3. Real geography

Admin-1 regions are converted to disjoint H3 cells (resolution chosen per country, ≤ ~12,000
cells); the 40 largest Natural Earth populated places become cities; each region gets a
"rest of region" settlement. Population is allocated to cities by metropolitan population within
the WDI urban share, and the rest by region area. This is a crude, documented downscaling;
gridded population (WorldPop/GHSL) is the next upgrade.

## 4. Economy calibration (M4, first step)

Parameters with provenance (see each country report's table): population, unemployment, government
consumption share, urban share (data); participation rescaled to the model's working-age band,
productivity level and trend, natural unemployment, CO2 intensity and its trend (inferred);
pension age and inflation target (assumed, `countries/institutions.toml`). The **effective income
tax rate is calibrated** by bisection so that simulated unemployment matches the ILO rate
(all six converge; resulting rates: Japan 0.44, Germany 0.42, Brazil 0.33, USA 0.21,
India 0.12, Nigeria 0.11 — ordered like real tax burdens, which was not targeted).

**Untargeted moments vs WDI (honest summary):**

| moment | finding |
|---|---|
| Real GDP growth | close where productivity trend drives growth (USA 2.3% vs 2.5%; India 4.8% vs 6.0%); too low for Japan/Germany (no capital deepening) |
| CPI inflation | anchored near the assumed inflation target; misses high-inflation episodes (Brazil, Nigeria) — no exchange rate, supply shocks or fiscal dominance |
| Household consumption share | 80–90% in the model vs 53–68% observed: no investment or foreign sector (structural) |
| Income Gini | too low for unequal countries (Brazil 0.31 vs 0.52): no within-record heterogeneity, no capital-income concentration, no informality |

## 5. Climate module (Phase 3 start)

FaIR-v1-type carbon cycle (Millar et al. 2017) → CO2 forcing (Myhre et al. 1998) + AR6 non-CO2
forcing → two-layer energy balance model (Geoffroy et al. 2013), run from 1750.
Calibration with held-out tests:

* r0 fitted on Mauna Loa CO2 1959–1999 → **out-of-sample RMSE 1.6 ppm (2000–2023)**; 2023: 424 vs 421 ppm.
* Climate feedback λ fitted on 1850–1989 gives ECS 6.6 K and fails out of sample (0.56 K RMSE),
  so the CMIP5-range prior λ = 1.13 (ECS 3.3 K) is kept: **0.18 K out-of-sample RMSE**; the model
  runs ~0.2 K warm over 2014–2023.

Country economies now account territorial CO2 (`EmissionsEngine`): intensity per unit of real
output and its decline rate inferred from GCB emissions and WDI GDP; base-year emissions are within
~8% of observed (the residual equals the model's GDP shortfall). Scenario levers:
`climate.decarbonisation_rate`, `climate.emissions_intensity`. No damages feed back to the economy yet.

## 6. Commands

```bash
python -m worldos data verify
python -m worldos country run IND --months 60
python -m worldos country validate DEU
python -m worldos climate validate
python -m worldos climate scenario --net-zero 2060
```
