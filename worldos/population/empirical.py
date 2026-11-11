"""Empirical (data-driven) demographic schedules and time-varying rate providers.

These classes expose the same interface as the parametric ``MortalitySchedule`` /
``FertilitySchedule`` so the demography engine and cohort model can run on either.

* Mortality: single-year central death rates m(x) from a life table (e.g. UN WPP), used as a
  constant hazard within each year of age; m(100+) applies up to ``max_age``.
* Fertility: single-year age-specific fertility rates (births per woman per year).
* Migration: net migrants by sex and single year of age per calendar year.
  - Estimation years: residual method, M(t, a+1) = P(t+1, a+1) - P(t, a) * L(a+1)/L(a),
    M(t, 0) = P(t+1, 0) - B(t) * L(0)/l(0), using the same life table (standard demographic
    practice; residuals also absorb any inconsistencies in the source data).
  - Projection years: total net migration from the source, distributed with the Rogers-Castro
    (1981) model migration schedule (labour-force peak around age 20-25), split equally by sex.
* Mortality improvement beyond the last life table: m(x) * exp(-k), with k solved so that
  period life expectancy matches the source's projected value (a proportional shift in the
  spirit of Lee & Carter 1992, with a single index).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from worldos.population.schedules import FEMALE, MALE

N_SINGLE = 101  # ages 0..100 (100 = open interval)


@dataclass(frozen=True)
class EmpiricalMortality:
    mx_female: np.ndarray  # length 101
    mx_male: np.ndarray
    max_age: int = 110

    @property
    def n_ages(self) -> int:
        return self.max_age * 12 + 1

    def _mx(self, sex) -> np.ndarray:
        return np.where(np.asarray(sex) == MALE, 1, 0)

    def hazard(self, age_years: np.ndarray, sex: np.ndarray | int) -> np.ndarray:
        idx = np.clip(np.floor(np.asarray(age_years)).astype(int), 0, N_SINGLE - 1)
        return np.where(np.asarray(sex) == MALE, self.mx_male[idx], self.mx_female[idx])

    def monthly_death_prob(self, age_months: np.ndarray, sex: np.ndarray | int) -> np.ndarray:
        age_months = np.asarray(age_months)
        q = -np.expm1(-self.hazard(age_months / 12.0, sex) / 12.0)
        return np.where(age_months >= self.max_age * 12, 1.0, q)

    def survivorship(self, sex: int) -> np.ndarray:
        ages = np.arange(self.n_ages)
        q = self.monthly_death_prob(ages, sex)
        surv = np.ones(self.n_ages)
        surv[1:] = np.cumprod(1.0 - q[1:])
        return surv

    def life_expectancy(self, sex: int) -> float:
        """Period life expectancy at birth (constant hazard within each year of age)."""
        mx = self.mx_male if sex == MALE else self.mx_female
        ages = np.arange(self.max_age)
        m = mx[np.clip(ages, 0, N_SINGLE - 1)]
        surv_start = np.exp(-np.concatenate([[0.0], np.cumsum(m)[:-1]]))
        lived = np.where(m > 0, surv_start * (1 - np.exp(-m)) / np.maximum(m, 1e-12), surv_start)
        return float(lived.sum())

    def improved(self, k_female: float, k_male: float) -> "EmpiricalMortality":
        return EmpiricalMortality(self.mx_female * np.exp(-k_female), self.mx_male * np.exp(-k_male), self.max_age)


def shift_to_life_expectancy(base: EmpiricalMortality, e0_female: float, e0_male: float) -> EmpiricalMortality:
    def solve(sex: int, target: float) -> float:
        f = lambda k: (base.improved(k, 0) if sex == FEMALE else base.improved(0, k)).life_expectancy(sex) - target
        lo, hi = -2.0, 3.0
        if f(lo) * f(hi) > 0:
            return 0.0
        return brentq(f, lo, hi, xtol=1e-8)

    return base.improved(solve(FEMALE, e0_female), solve(MALE, e0_male))


@dataclass(frozen=True)
class EmpiricalFertility:
    asfr_by_age: np.ndarray  # length 101, births per woman per year
    sex_ratio_at_birth: float = 1.05
    age_min: int = 15
    age_max: int = 50

    @property
    def tfr(self) -> float:
        return float(self.asfr_by_age.sum())

    def asfr(self, age_years: np.ndarray) -> np.ndarray:
        idx = np.clip(np.floor(np.asarray(age_years)).astype(int), 0, N_SINGLE - 1)
        return self.asfr_by_age[idx]

    def monthly_birth_prob(self, age_months: np.ndarray) -> np.ndarray:
        return self.asfr(np.asarray(age_months) / 12.0) / 12.0

    @property
    def female_birth_share(self) -> float:
        return 1.0 / (1.0 + self.sex_ratio_at_birth)

    def mean_age_at_childbearing(self) -> float:
        a = np.arange(N_SINGLE) + 0.5
        return float((a * self.asfr_by_age).sum() / max(self.asfr_by_age.sum(), 1e-12))


def rogers_castro(ages: np.ndarray, a1=0.02, alpha1=0.1, a2=0.06, alpha2=0.1, mu2=20.0, lambda2=0.4, c=0.003) -> np.ndarray:
    """Rogers & Castro (1981) seven-parameter model migration schedule (unnormalised)."""
    x = np.asarray(ages, float)
    return a1 * np.exp(-alpha1 * x) + a2 * np.exp(-alpha2 * (x - mu2) - np.exp(-lambda2 * (x - mu2))) + c


@dataclass
class YearRates:
    mortality: EmpiricalMortality
    fertility: EmpiricalFertility
    migration_female: np.ndarray  # net migrants (persons) by single age, for the calendar year
    migration_male: np.ndarray
    source: str  # "estimate" | "projection"


@dataclass
class DemographicRates:
    """Calendar-year demographic rates for one country, with documented extrapolation rules."""

    iso3: str
    years: dict[int, YearRates] = field(default_factory=dict)

    def for_year(self, year: int) -> YearRates:
        if year in self.years:
            return self.years[year]
        known = sorted(self.years)
        return self.years[known[-1] if year > known[-1] else known[0]]


def residual_migration(pop_now: np.ndarray, pop_next: np.ndarray, Lx: np.ndarray, l0: float, births: float) -> np.ndarray:
    """Residual net migration by single age (0..100) for one sex and one year."""
    surv = np.ones(N_SINGLE)
    surv[:-1] = Lx[1:] / np.maximum(Lx[:-1], 1e-12)
    m = np.zeros(N_SINGLE)
    m[0] = pop_next[0] - births * Lx[0] / l0
    m[1:100] = pop_next[1:100] - pop_now[0:99] * surv[0:99]
    # open interval 100+: survivors of 99 and of 100+ (survival of 100+ approximated by surv[99])
    m[100] = pop_next[100] - (pop_now[99] + pop_now[100]) * surv[99]
    return m
