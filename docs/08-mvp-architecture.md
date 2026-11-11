# 08 · MVP Architecture and Repository Structure (Task 8)

## 1. MVP definition (WORLD-OS Alpha)

One country (synthetic in Alpha, real in Alpha+1), multiple regions and cities, 10⁵–10⁶ weighted
person records, 10²–10³ firms, a bank, a central bank and a government, monthly ticks, scenario
experiments with uncertainty, and a validation suite. All of this is implemented now, except that
the country is synthetic.

## 2. Repository layout (as built)

```
worldos/
  config.py              typed, validated configuration (pydantic), overrides by dotted path
  build.py               assemble World + default engine set from config and seed
  cli.py, __main__.py    `python -m worldos run|experiment|validate|catalogue`
  core/
    clock.py             SimDate, SimulationClock, EventScheduler
    rng.py               named reproducible random streams
    table.py             ColumnTable (struct-of-arrays agent storage)
    provenance.py        ValueKind, Provenance, Uncertainty
    actions.py           Action, Message, ActionKernel (validation + audit)
    engine.py            Engine contract
    world.py             World (all mutable state; clone())
    simulation.py        Simulation kernel (events, engines, invariants)
  geo/                   GeoHierarchy (H3), synthetic country builder
  population/            schedules (Siler, Hadwiger, Lotka), synthesis, demography engine, cohort model
  economy/               state (firms, bank, gov, CB, flows), ledger services, engines, initialisation
  institutions/          policies (Taylor rule, fiscal rule), kernel registrations, policy engines
  scenario/              experiment specs, runner, statistics, markdown report
  validation/            metrics, invariants, statistics, validation suite
  data/                  catalogue (TOML + loader), observation store and ingestion
  report/                self-contained HTML dashboard
tests/                   pytest suite (unit, integration, statistical)
examples/experiments/    TOML experiment specifications
docs/                    specifications (this folder)
```

## 3. Dependency rules

`core` depends on nothing internal except `config`, and on `validation.invariants` inside
`simulation.py`. Domain packages (`geo`, `population`, `economy`, `institutions`) depend on `core`.
`scenario`, `report` and `cli` sit on top. Engines never import each other's state, only shared
helpers.

## 4. Interfaces for extension

* **New engine:** subclass `Engine`, choose `order`/`period_months`, read and write only the state
  it owns, add invariants for its conservation laws, register it in `build.default_engines()`.
* **New strategic agent:** implement `decide(observation) -> Action | None`; register the action
  kind with validators and allowed actors in `institutions/actions.py`.
* **New data source:** add a catalogue entry; write a connector that emits `Observation`s with
  provenance; never write observations into simulation state directly — calibration code maps
  observations to config.

## 5. Planned additions for Alpha+1

`api/` (FastAPI service), `store/` (Parquet snapshots + run manifests), `connectors/` (WPP, WDI),
`calibration/` (moment matching, SBI), `ui/` (React command centre).
