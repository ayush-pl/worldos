"""Assemble a complete Alpha world and its engine set from a configuration and a seed."""

from __future__ import annotations

from worldos.config import WorldConfig
from worldos.core.actions import ActionKernel
from worldos.core.clock import EventScheduler, SimDate, SimulationClock
from worldos.core.rng import RandomStreams
from worldos.core.world import World
from worldos.climate.emissions import EmissionsEngine
from worldos.economy.engines import (
    CreditEngine, FirmFinanceEngine, FirmPlanningEngine, GoodsMarketEngine, HouseholdExpectationsEngine,
    IncomeEngine, LabourMarketEngine, ProductionEngine, TechnologyEngine,
)
from worldos.economy.initialize import initialize_economy
from worldos.economy.state import BankState, CentralBankState, FirmTable, GovernmentState
from worldos.geo.synthetic import build_synthetic_geography
from worldos.institutions.actions import register_default_actions
from worldos.institutions.engines import CentralBankEngine, GovernmentPolicyEngine
from worldos.institutions.policies import FiscalRulePolicy, TaylorRulePolicy
from worldos.population.demography import DemographyEngine
from worldos.population.schedules import FertilitySchedule, MortalitySchedule, stable_population
from worldos.population.synthesis import synthesize_population
from worldos.validation.metrics import MetricsEngine


def build_world(config: WorldConfig | None = None, seed: int = 0) -> World:
    config = (config or WorldConfig()).model_copy(deep=True)
    rng = RandomStreams(seed)
    geo, layout = build_synthetic_geography(config.geo, rng.get("init.geo"))
    mort = MortalitySchedule.from_config(config.demography)
    fert = FertilitySchedule.from_config(config.demography)
    stable = stable_population(mort, fert)
    persons, households = synthesize_population(config.population, stable.female, stable.male, layout,
                                                rng.get("init.population"))
    kernel = ActionKernel()
    register_default_actions(kernel)
    world = World(
        config=config, clock=SimulationClock(SimDate.parse(config.simulation.start)), rng=rng, geo=geo,
        layout=layout, mortality=mort, fertility=fert, persons=persons, households=households,
        firms=FirmTable(), bank=BankState(), government=GovernmentState(), central_bank=CentralBankState(),
        kernel=kernel, events=EventScheduler(), run_id=f"seed{seed}",
    )
    initialize_economy(world, rng.get("init.economy"))
    world.strategic_agents = {
        "central_bank": TaylorRulePolicy(config.central_bank),
        "government": FiscalRulePolicy(config.government, inflation_target=config.central_bank.inflation_target),
    }
    neutral_debt_target(world)
    return world


def neutral_debt_target(world) -> float:
    """Set the fiscal rule's debt target to the initial debt ratio, so the rule starts neutral."""
    f = world.firms
    annual_gdp = 12 * float((f.price * f.exp_demand)[f.alive].sum())
    ratio = world.government.debt / annual_gdp
    world.config.government.debt_target = round(ratio, 4)
    world.strategic_agents["government"].cfg = world.config.government
    return ratio


def default_engines() -> list:
    return [
        DemographyEngine(), TechnologyEngine(), FirmPlanningEngine(), CreditEngine(), LabourMarketEngine(),
        ProductionEngine(), EmissionsEngine(), IncomeEngine(), GoodsMarketEngine(), FirmFinanceEngine(),
        HouseholdExpectationsEngine(), CentralBankEngine(), GovernmentPolicyEngine(), MetricsEngine(),
    ]
