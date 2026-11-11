# 02 · Gap Analysis Against Existing Projects (Task 2)

**Status:** v0.1 · Descriptions are high-level and should be re-checked against each project's
current publications before external use.

## 1. Comparison

| Project | What it is | Strengths | Gaps relative to WORLD-OS |
|---|---|---|---|
| **Destination Earth (DestinE)** | EU initiative (implemented with ECMWF, ESA and EUMETSAT) building high-resolution digital twins of the Earth system on EuroHPC supercomputers; first twins target weather-induced extremes and climate-change adaptation | Physical fidelity (km-scale), operational data pipelines, HPC engineering | Socioeconomic and behavioural components are not its core; no agent-based economy or institutions |
| **Earth4All** | Club of Rome-associated initiative and system-dynamics model (successor to the *Limits to Growth*/World3 tradition), used for the 2022 book *Earth for All*; compares scenarios such as "Too Little Too Late" vs "Giant Leap" | Integrated socio-economic-environmental feedbacks; transparent stock-flow structure; long horizons | Aggregate regions, no heterogeneous agents, limited spatial detail, no explicit financial accounting at agent level |
| **EURACE** | EU-funded project (mid-2000s) that built a large agent-based model of the European economy on the FLAME framework; continued as Eurace@Unibi | Detailed agent-based macro with labour, goods, credit and financial markets; policy experiments | Economy only; no physical Earth system; limited demographic life-course detail |
| **NASA Earth System Digital Twins** | Research and technology programme (within NASA's Earth Science Technology Office) developing digital-twin capabilities for Earth observation and modelling | Earth observation integration, physical models, data systems | Physical/observational focus; human systems are external drivers |
| **Sugarscape** | Epstein & Axtell's *Growing Artificial Societies* (1996): stylised agents on a resource grid | Shows emergence of distribution, trade and migration from simple rules; pedagogical clarity | Not empirically grounded; no institutions, money or real geography |
| *Also relevant:* data-driven macro ABMs (e.g. Poledna et al., *European Economic Review* 2023, an ABM of Austria calibrated to national accounts and used for forecasting) | Shows ABMs can be competitive with standard macro models when fed rich micro data | Single economy; no environment |
| *Also relevant:* Integrated Assessment Models (GCAM, MESSAGEix, REMIND, IMAGE) | Energy-land-climate pathways, IPCC scenario work | Optimisation/equilibrium aggregates; no agent heterogeneity or institutions-as-agents |

## 2. What WORLD-OS adds (the defensible niche)

1. **Agent-level human systems coupled to a geography and to physical models**: DestinE-class
   physics is consumed as data or emulators; the novelty is in the human/institutional side.
2. **Strict accounting across all couplings**: SFC balance sheets for money, conservation for goods,
   energy, water and carbon, checked every tick.
3. **Provenance-first data twin**: every value labelled observed/estimated/inferred/simulated/forecast/assumed.
4. **Experimentation as the primary product**: branchable world states, ensembles with common
   random numbers, paired comparisons, standard reports.

## 3. What WORLD-OS should *reuse rather than rebuild*

| Need | Reuse candidate | Licence (verify) |
|---|---|---|
| Climate response to emissions | FaIR simple climate model (open source) | Apache-2.0 |
| Spatial index | H3 | Apache-2.0 (used in Alpha) |
| ABM tooling / visualisation | Mesa (prototyping), custom runtime for scale | Apache-2.0 |
| Crop yields | Published crop-model emulators | per project |
| Scenario drivers | SSP database (population, GDP, emissions) | IIASA terms |

## 4. Gaps that are research problems for WORLD-OS too

* Empirical validation of a coupled ABM at national scale (EURACE and successors show this is hard).
* Calibration of thousands of behavioural parameters; requires simulation-based inference and
  surrogate models.
* Legitimate representation of political decision-making (docs/05 §5) without pretending to predict
  real governments.
