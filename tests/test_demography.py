import numpy as np
import pytest

from worldos.build import build_world
from worldos.config import DemographyConfig, WorldConfig, apply_overrides
from worldos.population.demography import CohortProjection, DemographyEngine, aggregate_age_sex
from worldos.population.schedules import FEMALE, MALE, FertilitySchedule, MortalitySchedule, stable_population
from worldos.population.tables import CHILD, RETIRED


@pytest.fixture
def schedules():
    c = DemographyConfig()
    return MortalitySchedule.from_config(c), FertilitySchedule.from_config(c)


def test_life_expectancy_plausible_and_male_lower(schedules):
    mort, _ = schedules
    e_f, e_m = mort.life_expectancy(FEMALE), mort.life_expectancy(MALE)
    assert 70 < e_m < e_f < 90


def test_fertility_schedule_integrates_to_tfr(schedules):
    _, fert = schedules
    ages = np.arange(0, 110 * 12)
    assert fert.monthly_birth_prob(ages).sum() == pytest.approx(fert.tfr, rel=1e-9)
    assert 25 < fert.mean_age_at_childbearing() < 33


def test_lotka_equation_solved(schedules):
    mort, fert = schedules
    sp = stable_population(mort, fert)
    ages = np.arange(mort.n_ages)
    lhs = (np.exp(-sp.r * ages / 12) * mort.survivorship(FEMALE) * fert.monthly_birth_prob(ages)
           * fert.female_birth_share).sum()
    assert lhs == pytest.approx(1.0, abs=1e-9)
    assert sp.total.sum() == pytest.approx(1.0)


def test_cohort_projection_is_expectation_of_microsimulation():
    cfg = apply_overrides(WorldConfig(), {"population.n_person_records": 30000})
    totals, cohort_total = [], None
    for seed in (1, 2, 3):
        w = build_world(cfg, seed)
        cohort = CohortProjection.from_persons(w.persons, w.mortality, w.fertility)
        engine = DemographyEngine()
        for _ in range(120):
            w.begin_tick()
            engine.step(w)
            w.clock.advance()
            cohort.step()
        f, m = aggregate_age_sex(w.persons, w.mortality.n_ages)
        totals.append((f.sum() + m.sum()) / cohort.total - 1)
    assert abs(np.mean(totals)) < 0.01


def test_life_course_and_households_consistent(small_config):
    w = build_world(small_config, 4)
    engine = DemographyEngine()
    for _ in range(36):
        w.begin_tick()
        engine.step(w)
        w.clock.advance()
    p, h = w.persons, w.households
    alive = p.alive
    assert np.all(h.alive[p.hh[alive]])
    assert not np.any(alive & (p.status == RETIRED) & (p.age_m < 65 * 12))
    assert not np.any(alive & (p.status == CHILD) & (p.age_m > 16 * 12))
    assert h.equity_share[h.alive].sum() == pytest.approx(1.0)
    births = (p.birth_tick >= 0).sum()
    assert births > 0


def test_every_household_has_an_adult_at_creation(small_config):
    w = build_world(small_config, 5)
    p, h = w.persons, w.households
    adult_count = np.bincount(p.hh[p.age_m >= 18 * 12], minlength=len(h))
    assert np.all(adult_count[h.alive] >= 1)
