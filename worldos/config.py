"""Typed, validated configuration for a WORLD-OS Alpha world.

Every numeric default below is a documented MODELLING ASSUMPTION for the
synthetic country "Synthland" (provenance kind ``ASSUMED``). They were chosen
to be of plausible order of magnitude for a middle/high-income economy and to
give a stable baseline; they are NOT calibrated to any real country. The fiscal parameters were tuned
(docs/06, "Baseline calibration") so that the stylised baseline has
unemployment of roughly 6-8%, low positive inflation and a stable public debt
ratio; this is calibration to stylised targets, not statistical estimation. Replacing
them with values estimated from the data catalogue is milestone M2 in
docs/09-implementation-roadmap.md.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SimulationConfig(_Model):
    start: str = "2025-01"
    strict_invariants: bool = True
    invariant_rel_tol: float = 1e-8


class GeoConfig(_Model):
    country_name: str = "Synthland (synthetic)"
    # Outer ring (lat, lng). Default lies over the open central Pacific so the
    # synthetic country cannot be mistaken for a real territory.
    polygon: list[tuple[float, float]] = Field(
        default_factory=lambda: [(1.0, -152.0), (1.0, -146.0), (5.0, -145.0), (6.5, -149.0), (4.5, -153.0)]
    )
    country_resolution: int = Field(5, ge=0, le=15)
    city_resolution: int = Field(7, ge=0, le=15)
    n_regions: int = Field(4, ge=1)
    n_cities: int = Field(8, ge=1)
    zipf_exponent: float = Field(1.0, gt=0)  # rank-size rule (Zipf 1949; Gabaix 1999)


class PopulationConfig(_Model):
    represented_population: float = Field(1_000_000, gt=0)
    n_person_records: int = Field(20_000, ge=100)
    partnership_rate: float = Field(0.6, ge=0, le=1)
    education_distribution: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.25])
    # P(child education | parent education); rows = parent level.
    education_mobility: list[list[float]] = Field(
        default_factory=lambda: [[0.45, 0.45, 0.10], [0.15, 0.60, 0.25], [0.05, 0.40, 0.55]]
    )
    school_leaving_age: list[int] = Field(default_factory=lambda: [16, 18, 22])
    participation_rate: float = Field(0.8, ge=0, le=1)
    pension_age: int = Field(65, ge=40, le=90)
    household_leave_rate: float = Field(0.10, ge=0, le=1)  # annual, adults 20-34 not heading a household
    wealth_pareto_alpha: float = Field(1.5, gt=1.0)

    @field_validator("education_distribution")
    @classmethod
    def _dist(cls, v: list[float]) -> list[float]:
        if len(v) != 3 or abs(sum(v) - 1) > 1e-9 or min(v) < 0:
            raise ValueError("education_distribution must be 3 non-negative shares summing to 1")
        return v

    @field_validator("education_mobility")
    @classmethod
    def _mob(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(r) != 3 or abs(sum(r) - 1) > 1e-9 or min(r) < 0 for r in v):
            raise ValueError("education_mobility must be a 3x3 row-stochastic matrix")
        return v


class DemographyConfig(_Model):
    # Siler (1979) competing-hazards mortality: mu(x) = a1 e^{-b1 x} + a2 + a3 e^{b3 x}
    siler_a1: float = Field(0.012, ge=0)
    siler_b1: float = Field(2.0, gt=0)
    siler_a2: float = Field(0.0004, ge=0)
    siler_a3: float = Field(3.0e-5, gt=0)
    siler_b3: float = Field(0.095, gt=0)
    male_mortality_multiplier: float = Field(1.25, gt=0)
    max_age: int = Field(110, ge=80, le=130)
    # Hadwiger fertility schedule (Hadwiger 1940; Chandola, Coleman & Hiorns 1999)
    total_fertility_rate: float = Field(2.05, ge=0)
    hadwiger_R: float = Field(28.5, gt=10)
    hadwiger_H: float = Field(3.6, gt=0)
    fertile_age_min: int = 15
    fertile_age_max: int = 50
    sex_ratio_at_birth: float = Field(1.05, gt=0)  # males per female


class EconomyConfig(_Model):
    n_firms: int = Field(200, ge=2)
    firm_size_sigma: float = Field(1.0, ge=0)  # lognormal dispersion of initial firm size
    initial_wage: float = Field(1000.0, gt=0)  # SYN$ per effective worker-month
    initial_productivity: float = Field(1000.0, gt=0)  # goods per effective worker-month
    initial_markup: float = Field(0.10, ge=0)
    education_productivity: list[float] = Field(default_factory=lambda: [0.75, 1.0, 1.5])
    initial_unemployment: float = Field(0.06, ge=0, lt=1)
    inventory_buffer: float = Field(0.10, ge=0)
    expectation_smoothing: float = Field(0.5, gt=0, le=1)
    quantity_step: float = Field(0.05, ge=0)
    markup_step: float = Field(0.02, ge=0)  # max monthly absolute markup adjustment
    wage_step: float = Field(0.01, ge=0)
    wage_indexation: float = Field(0.5, ge=0, le=1)  # share of past positive CPI inflation passed into wage offers
    wage_productivity_passthrough: float = Field(1.0, ge=0, le=1.5)  # share of trend productivity growth
    downward_wage_flexibility: float = Field(0.2, ge=0, le=1)  # scales cuts relative to rises (nominal rigidity)
    markup_bounds: tuple[float, float] = (0.0, 0.5)
    separation_rate: float = Field(0.01, ge=0, le=1)  # monthly exogenous job separations
    firing_tolerance: float = Field(0.05, ge=0)
    search_breadth: int = Field(3, ge=1)  # firms sampled per search (goods and jobs)
    market_rounds: int = Field(3, ge=1)
    consumption_income_propensity: float = Field(0.9, ge=0, le=1.5)
    consumption_wealth_propensity: float = Field(0.02, ge=0, le=1)  # per month
    income_expectation_weight: float = Field(0.25, gt=0, le=1)
    dividend_payout: float = Field(0.9, ge=0, le=1)
    liquidity_buffer_months: float = Field(1.0, ge=0)
    productivity_growth: float = Field(0.015, ge=-0.1, le=0.2)  # annual, exogenous in Alpha
    productivity_noise: float = Field(0.002, ge=0)
    entry_delay_months: int = Field(3, ge=0)
    initial_household_deposit_months: float = Field(5.0, ge=0)
    initial_firm_deposit_months: float = Field(1.5, ge=0)
    initial_firm_loan_months: float = Field(1.0, ge=0)


class BankConfig(_Model):
    capital_requirement: float = Field(0.08, gt=0, lt=1)  # Basel-style minimum, risk weight 1 in Alpha
    equity_target: float = Field(0.12, gt=0, lt=1)
    initial_equity_ratio: float = Field(0.12, gt=0, lt=1)
    loan_spread: float = Field(0.02, ge=0)
    risk_premium: float = Field(0.03, ge=0)
    deposit_markdown: float = Field(0.01, ge=0)
    amortization_months: int = Field(36, ge=1)
    max_debt_to_sales_months: float = Field(6.0, gt=0)

    @model_validator(mode="after")
    def _order(self) -> "BankConfig":
        if self.equity_target < self.capital_requirement:
            raise ValueError("equity_target must be >= capital_requirement")
        return self


class GovernmentConfig(_Model):
    income_tax_rate: float = Field(0.34, ge=0, le=0.9)  # taxes + social contributions on wages
    tax_allowance_ratio: float = Field(0.1, ge=0)  # tax-free allowance as share of mean wage
    corporate_tax_rate: float = Field(0.2, ge=0, le=0.9)
    unemployment_replacement: float = Field(0.5, ge=0, le=1.5)
    pension_replacement: float = Field(0.40, ge=0, le=1.5)
    spending_share: float = Field(0.13, ge=0, le=0.8)  # public consumption / trailing GDP
    debt_target: float = Field(0.55, ge=0)
    fiscal_adjustment: float = Field(0.05, ge=0)
    max_tax_change: float = Field(0.02, ge=0)
    tax_rate_bounds: tuple[float, float] = (0.05, 0.6)


class CentralBankConfig(_Model):
    initial_policy_rate: float = Field(0.02, ge=0)
    neutral_real_rate: float = 0.01
    inflation_target: float = 0.02
    phi_inflation: float = Field(0.5, ge=0)
    phi_unemployment: float = Field(0.5, ge=0)
    natural_unemployment: float = Field(0.06, ge=0, lt=1)
    max_step: float = Field(0.0075, ge=0)
    rate_bounds: tuple[float, float] = (0.0, 0.20)


class WorldConfig(_Model):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    geo: GeoConfig = Field(default_factory=GeoConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    demography: DemographyConfig = Field(default_factory=DemographyConfig)
    economy: EconomyConfig = Field(default_factory=EconomyConfig)
    bank: BankConfig = Field(default_factory=BankConfig)
    government: GovernmentConfig = Field(default_factory=GovernmentConfig)
    central_bank: CentralBankConfig = Field(default_factory=CentralBankConfig)


def set_path(obj: Any, path: str, value: Any) -> None:
    """Set a dotted attribute path (``government.income_tax_rate``) with validation on assignment."""
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise AttributeError(f"unknown path {path!r}")
    setattr(target, parts[-1], value)


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def apply_overrides(config: WorldConfig, overrides: dict[str, Any]) -> WorldConfig:
    new = config.model_copy(deep=True)
    for path, value in overrides.items():
        set_path(new, path, value)
    return new


def load_config(path: str | None = None) -> WorldConfig:
    """Load a TOML config (sections mirror WorldConfig); missing keys take defaults."""
    if path is None:
        return WorldConfig()
    import tomllib

    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return WorldConfig.model_validate(data)
