"""Demographic microsimulation engine and its aggregate (cohort-component) twin.

Monthly order of events for every person record (identical in both models):
  1. ageing by one month;
  2. death with probability q(age, sex) (Siler hazard);
  3. for surviving women of fertile age, a birth with probability f(age)/12;
  4. life-course transitions: school -> labour market at the school-leaving
     age of the person's education level, retirement at the pension age;
  5. household dynamics: young adults leave home to form new households;
     households whose last member died pass their estate to an heir household.

``CohortProjection`` applies exactly the same rates to expected counts by
single month of age. It is (a) the aggregate representation level of the
population and (b) the exact expectation of the microsimulation's population
counts, which makes the micro <-> aggregate consistency testable
(tests/test_demography.py, worldos validate).
"""

from __future__ import annotations

import numpy as np

from worldos.core.engine import Engine
from worldos.population.schedules import FEMALE, MALE, FertilitySchedule, MortalitySchedule
from worldos.population.tables import (
    CHILD, EMPLOYED, INACTIVE, RETIRED, STUDENT, UNEMPLOYED, recount_members,
)


class DemographyEngine(Engine):
    name = "demography"
    order = 10
    period_months = 1

    def step(self, world) -> None:
        p, h, fl = world.persons, world.households, world.flows
        pc = world.config.population
        rng = world.rng.get(self.name)
        mort: MortalitySchedule = world.mortality
        fert: FertilitySchedule = world.fertility

        alive = np.flatnonzero(p.alive)
        p.age_m[alive] += 1

        # 2. mortality
        q = mort.monthly_death_prob(p.age_m[alive], p.sex[alive])
        die = alive[rng.random(len(alive)) < q]
        p.alive[die] = False
        p.employer[die] = -1
        fl.deaths = float(p.weight[die].sum())

        # 3. fertility
        alive = np.flatnonzero(p.alive)
        age = p.age_m[alive]
        women = alive[(p.sex[alive] == FEMALE) & (age >= fert.age_min * 12) & (age < fert.age_max * 12)]
        mothers = women[rng.random(len(women)) < fert.monthly_birth_prob(p.age_m[women])]
        if len(mothers):
            mobility = np.asarray(pc.education_mobility)
            u = rng.random(len(mothers))
            edu = (u[:, None] > np.cumsum(mobility[p.edu[mothers]], axis=1)[:, :-1]).sum(axis=1)
            sex = np.where(rng.random(len(mothers)) < fert.female_birth_share, FEMALE, MALE)
            p.add(len(mothers), hh=p.hh[mothers], sex=sex, edu=edu, age_m=0, status=CHILD,
                  weight=p.weight[mothers], birth_tick=world.clock.tick)
            fl.births = float(p.weight[mothers].sum())

        # 4. life-course transitions (exact month of crossing a threshold)
        alive = np.flatnonzero(p.alive)
        age, edu, st = p.age_m[alive], p.edu[alive], p.status[alive]
        leave = np.asarray(pc.school_leaving_age)[edu] * 12
        to_student = alive[(st == CHILD) & (age == 16 * 12) & (leave > 16 * 12)]
        p.status[to_student] = STUDENT
        entering = alive[((st == CHILD) | (st == STUDENT)) & (age >= leave) & (age < pc.pension_age * 12)]
        part = rng.random(len(entering)) < pc.participation_rate
        p.status[entering] = np.where(part, UNEMPLOYED, INACTIVE)
        retiring = alive[(age >= pc.pension_age * 12) & (st != RETIRED)]
        p.status[retiring] = RETIRED
        p.employer[retiring] = -1

        # 5. household dynamics
        self._leave_home(world, rng)
        self._settle_estates(world, rng)

    @staticmethod
    def _leave_home(world, rng: np.random.Generator) -> None:
        p, h = world.persons, world.households
        rate = world.config.population.household_leave_rate
        if rate <= 0:
            return
        recount_members(p, h)
        alive = np.flatnonzero(p.alive)
        oldest = np.zeros(len(h), dtype=np.int64)
        np.maximum.at(oldest, p.hh[alive], p.age_m[alive])
        age = p.age_m[alive]
        cand = alive[(age >= 20 * 12) & (age < 35 * 12) & (age < oldest[p.hh[alive]])
                     & np.isin(p.status[alive], (EMPLOYED, UNEMPLOYED, INACTIVE))]
        monthly = 1.0 - (1.0 - rate) ** (1.0 / 12.0)
        movers = cand[rng.random(len(cand)) < monthly]
        if not len(movers):
            return
        parents = p.hh[movers]
        frac = 1.0 / h.n_members[parents]
        dep, eq, inc = h.deposits[parents] * frac, h.equity_share[parents] * frac, h.income_exp[parents] * frac
        np.subtract.at(h.deposits, parents, dep)
        np.subtract.at(h.equity_share, parents, eq)
        np.subtract.at(h.income_exp, parents, inc)
        new = h.add(len(movers), region=h.region[parents], city=h.city[parents], weight=h.weight[parents],
                    deposits=dep, equity_share=eq, income_exp=inc)
        p.hh[movers] = new
        recount_members(p, h)

    @staticmethod
    def _settle_estates(world, rng: np.random.Generator) -> None:
        p, h = world.persons, world.households
        recount_members(p, h)
        empty = np.flatnonzero(h.alive & (h.n_members == 0))
        if not len(empty):
            return
        living = np.flatnonzero(h.alive & (h.n_members > 0))
        heirs = living[rng.integers(0, len(living), size=len(empty))]
        np.add.at(h.deposits, heirs, h.deposits[empty])
        np.add.at(h.loans, heirs, h.loans[empty])
        np.add.at(h.equity_share, heirs, h.equity_share[empty])
        h.deposits[empty] = 0.0
        h.loans[empty] = 0.0
        h.equity_share[empty] = 0.0
        h.income_exp[empty] = 0.0
        h.alive[empty] = False


class CohortProjection:
    """Aggregate population model: expected counts by sex and single month of age."""

    def __init__(self, female: np.ndarray, male: np.ndarray, mort: MortalitySchedule, fert: FertilitySchedule) -> None:
        if len(female) != mort.n_ages or len(male) != mort.n_ages:
            raise ValueError("count vectors must span 0..max_age in months")
        self.female = np.asarray(female, float).copy()
        self.male = np.asarray(male, float).copy()
        ages = np.arange(mort.n_ages)
        self._sf = 1.0 - mort.monthly_death_prob(ages, FEMALE)
        self._sm = 1.0 - mort.monthly_death_prob(ages, MALE)
        self._pb = fert.monthly_birth_prob(ages)
        self._fshare = fert.female_birth_share
        self.births_history: list[float] = []

    @classmethod
    def from_persons(cls, persons, mort: MortalitySchedule, fert: FertilitySchedule) -> "CohortProjection":
        """Aggregate a person table into the cohort representation (micro -> aggregate)."""
        f, m = aggregate_age_sex(persons, mort.n_ages)
        return cls(f, m, mort, fert)

    @property
    def total(self) -> float:
        return float(self.female.sum() + self.male.sum())

    def step(self) -> None:
        f = np.zeros_like(self.female)
        m = np.zeros_like(self.male)
        f[1:] = self.female[:-1] * self._sf[1:]
        m[1:] = self.male[:-1] * self._sm[1:]
        births = float((f * self._pb).sum())
        f[0] = births * self._fshare
        m[0] = births * (1.0 - self._fshare)
        self.female, self.male = f, m
        self.births_history.append(births)


def aggregate_age_sex(persons, n_ages: int) -> tuple[np.ndarray, np.ndarray]:
    alive = persons.alive
    age = np.minimum(persons.age_m[alive], n_ages - 1)
    sex, w = persons.sex[alive], persons.weight[alive]
    f = np.bincount(age[sex == FEMALE], weights=w[sex == FEMALE], minlength=n_ages)
    m = np.bincount(age[sex == MALE], weights=w[sex == MALE], minlength=n_ages)
    return f, m
