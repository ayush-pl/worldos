"""Demographic rate schedules and stable-population theory.

* Mortality: Siler (1979) competing-hazards model,
  mu(x) = a1*exp(-b1*x) + a2 + a3*exp(b3*x)  (infant / background / senescent).
* Fertility: Hadwiger (1940) function as parameterised by Chandola, Coleman &
  Hiorns (1999), numerically normalised so the schedule sums exactly to the TFR.
* Initial age structure: Lotka's stable population. Given the schedules, the
  intrinsic growth rate r solves  sum_a exp(-r a) l_f(a) m_f(a) = 1  and the
  stable age distribution is c(a) ~ exp(-r a) l(a). Using it means the
  synthetic population's starting pyramid is *implied by* the stated
  assumptions rather than invented separately.

All rates are evaluated on a monthly age grid, matching the monthly kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from worldos.config import DemographyConfig

FEMALE, MALE = 0, 1


@dataclass(frozen=True)
class MortalitySchedule:
    a1: float
    b1: float
    a2: float
    a3: float
    b3: float
    male_multiplier: float
    max_age: int

    @classmethod
    def from_config(cls, c: DemographyConfig) -> "MortalitySchedule":
        return cls(c.siler_a1, c.siler_b1, c.siler_a2, c.siler_a3, c.siler_b3, c.male_mortality_multiplier, c.max_age)

    @property
    def n_ages(self) -> int:
        return self.max_age * 12 + 1

    def hazard(self, age_years: np.ndarray, sex: np.ndarray | int) -> np.ndarray:
        x = np.asarray(age_years, dtype=float)
        mu = self.a1 * np.exp(-self.b1 * x) + self.a2 + self.a3 * np.exp(self.b3 * x)
        return mu * np.where(np.asarray(sex) == MALE, self.male_multiplier, 1.0)

    def monthly_death_prob(self, age_months: np.ndarray, sex: np.ndarray | int) -> np.ndarray:
        """Probability of dying during the month in which age (in months) is ``age_months``.

        Hazard evaluated at mid-month; certain death at ``max_age``.
        """
        age_months = np.asarray(age_months)
        mu = self.hazard((age_months + 0.5) / 12.0, sex)
        q = -np.expm1(-mu / 12.0)
        return np.where(age_months >= self.max_age * 12, 1.0, q)

    def survivorship(self, sex: int) -> np.ndarray:
        """l(a): probability that a newborn reaches the start of month-age a (a = 0..n_ages-1)."""
        ages = np.arange(self.n_ages)
        q = self.monthly_death_prob(ages, sex)
        surv = np.ones(self.n_ages)
        surv[1:] = np.cumprod(1.0 - q[1:])  # newborns are not exposed in their birth month (age 0)
        return surv

    def life_expectancy(self, sex: int) -> float:
        """Period life expectancy at birth in years under the microsimulation's timing convention."""
        l = self.survivorship(sex)
        q = self.monthly_death_prob(np.arange(self.n_ages), sex)
        # Months lived: newborn month counted fully; deaths assumed at mid-month.
        lived = l[1:] * (1.0 - 0.5 * q[1:])
        return float((1.0 + lived.sum()) / 12.0)


@dataclass(frozen=True)
class FertilitySchedule:
    tfr: float
    R: float
    H: float
    age_min: int
    age_max: int
    sex_ratio_at_birth: float

    @classmethod
    def from_config(cls, c: DemographyConfig) -> "FertilitySchedule":
        return cls(c.total_fertility_rate, c.hadwiger_R, c.hadwiger_H, c.fertile_age_min, c.fertile_age_max,
                   c.sex_ratio_at_birth)

    def _shape(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        safe = np.where(x > 0, x, 1.0)
        f = (self.R / safe) ** 1.5 * np.exp(-self.H**2 * (self.R / safe + safe / self.R - 2.0))
        return np.where((x >= self.age_min) & (x < self.age_max), f, 0.0)

    def _norm(self) -> float:
        ages = (np.arange(self.age_min * 12, self.age_max * 12) + 0.5) / 12.0
        return float(self._shape(ages).sum() / 12.0)

    def asfr(self, age_years: np.ndarray) -> np.ndarray:
        """Age-specific fertility rate (births per woman per year)."""
        n = self._norm()
        return np.zeros_like(np.asarray(age_years, float)) if n == 0 else self.tfr * self._shape(age_years) / n

    def monthly_birth_prob(self, age_months: np.ndarray) -> np.ndarray:
        return self.asfr((np.asarray(age_months) + 0.5) / 12.0) / 12.0

    @property
    def female_birth_share(self) -> float:
        return 1.0 / (1.0 + self.sex_ratio_at_birth)

    def mean_age_at_childbearing(self) -> float:
        ages = (np.arange(self.age_min * 12, self.age_max * 12) + 0.5) / 12.0
        f = self.asfr(ages)
        return float((ages * f).sum() / f.sum())


@dataclass(frozen=True)
class StablePopulation:
    r: float  # intrinsic annual growth rate
    female: np.ndarray  # share of total population by month-age
    male: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.female + self.male


def stable_population(mort: MortalitySchedule, fert: FertilitySchedule) -> StablePopulation:
    ages = np.arange(mort.n_ages)
    t = ages / 12.0
    lf = mort.survivorship(FEMALE)
    lm = mort.survivorship(MALE)
    phi = lf * fert.monthly_birth_prob(ages) * fert.female_birth_share  # net maternity, female births

    def lotka(r: float) -> float:
        return float((np.exp(-r * t) * phi).sum() - 1.0)

    r = 0.0 if phi.sum() == 0 else brentq(lotka, -0.2, 0.2)
    f = np.exp(-r * t) * lf
    m = fert.sex_ratio_at_birth * np.exp(-r * t) * lm
    total = f.sum() + m.sum()
    return StablePopulation(float(r), f / total, m / total)
