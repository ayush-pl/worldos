"""Synthetic population generation (privacy-preserving by construction).

Produces person and household records consistent with a target age-sex
distribution. No record corresponds to a real individual. For real countries
the target marginals come from UN WPP / census tables and the household
structure is fitted with iterative proportional fitting or combinatorial
optimisation (milestone M3, docs/09); Alpha uses the rule-based household
assembly below:

1. Sample (sex, age) records from the target distribution.
2. Assign education targets and activity status by age.
3. Pair a share of adult women (20-75) with men (20-80) by rank-matching on
   age (assortative), forming couple households; remaining adults live alone.
4. Attach each child (<18) to a random household whose oldest member is
   20-45 years older (fallback: nearest feasible household).
5. Place households: on H3 cells in proportion to gridded population when the layout has a gridded
   layer (real countries, Kontur Population), otherwise in settlements by their population shares.
"""

from __future__ import annotations

import numpy as np

from worldos.config import PopulationConfig
from worldos.geo.synthetic import SettlementLayout
from worldos.population.schedules import FEMALE, MALE
from worldos.population.tables import (
    CHILD, INACTIVE, RETIRED, STUDENT, UNEMPLOYED, HouseholdTable, PersonTable, recount_members,
)

ADULT_MONTHS = 18 * 12


def activity_status(age_m: np.ndarray, edu: np.ndarray, cfg: PopulationConfig, rng: np.random.Generator) -> np.ndarray:
    leave = np.asarray(cfg.school_leaving_age)[edu] * 12
    status = np.full(len(age_m), INACTIVE, dtype=np.int8)
    status[age_m < leave] = STUDENT
    status[age_m < 16 * 12] = CHILD
    working_age = (age_m >= leave) & (age_m < cfg.pension_age * 12)
    participates = rng.random(len(age_m)) < cfg.participation_rate
    status[working_age & participates] = UNEMPLOYED  # employment is assigned by the labour market
    status[age_m >= cfg.pension_age * 12] = RETIRED
    return status


def synthesize_population(
    cfg: PopulationConfig,
    female_dist: np.ndarray,
    male_dist: np.ndarray,
    layout: SettlementLayout,
    rng: np.random.Generator,
) -> tuple[PersonTable, HouseholdTable]:
    """``female_dist``/``male_dist``: non-negative weights by month-age (any scale)."""
    n = cfg.n_person_records
    weight = cfg.represented_population / n
    p = np.concatenate([female_dist, male_dist]).astype(float)
    p /= p.sum()
    n_ages = len(female_dist)
    draw = rng.choice(len(p), size=n, p=p)
    sex = (draw >= n_ages).astype(np.int8)
    age_m = (draw % n_ages).astype(np.int32)
    edu = rng.choice(3, size=n, p=cfg.education_distribution).astype(np.int8)

    persons = PersonTable(capacity=int(n * 1.5))
    pidx = persons.add(n, sex=sex, age_m=age_m, edu=edu, weight=weight,
                       status=activity_status(age_m, edu, cfg, rng), birth_tick=-age_m)

    hh_of = np.full(n, -1, dtype=np.int64)
    adult = age_m >= ADULT_MONTHS
    # --- couples by rank-matching on age
    f_cand = np.flatnonzero(adult & (sex == FEMALE) & (age_m >= 20 * 12) & (age_m < 75 * 12))
    f_cand = f_cand[rng.random(len(f_cand)) < cfg.partnership_rate]
    m_pool = np.flatnonzero(adult & (sex == MALE) & (age_m >= 20 * 12) & (age_m < 80 * 12))
    k = min(len(f_cand), len(m_pool))
    f_sel = rng.choice(f_cand, size=k, replace=False)
    m_sel = rng.choice(m_pool, size=k, replace=False)
    f_sel = f_sel[np.argsort(age_m[f_sel], kind="stable")]
    m_sel = m_sel[np.argsort(age_m[m_sel], kind="stable")]
    hh_of[f_sel] = np.arange(k)
    hh_of[m_sel] = np.arange(k)
    # --- singles
    singles = np.flatnonzero(adult & (hh_of < 0))
    hh_of[singles] = k + np.arange(len(singles))
    n_hh = k + len(singles)
    if n_hh == 0:
        raise ValueError("population has no adults; cannot form households")

    head_age = np.zeros(n_hh, dtype=np.int64)
    np.maximum.at(head_age, hh_of[adult], age_m[adult])
    # --- children: attach to households whose head is 20..45 years older
    order = np.argsort(head_age, kind="stable")
    sorted_age = head_age[order]
    children = np.flatnonzero(~adult)
    lo = np.searchsorted(sorted_age, age_m[children] + 20 * 12, side="left")
    hi = np.searchsorted(sorted_age, age_m[children] + 45 * 12, side="right")
    empty = hi <= lo
    # fallback: the nearest household whose head is at least 18 years older
    lo_fb = np.minimum(np.searchsorted(sorted_age, age_m[children] + 18 * 12, side="left"), n_hh - 1)
    pick = np.where(empty, lo_fb, lo + np.floor(rng.random(len(children)) * np.maximum(hi - lo, 1)).astype(np.int64))
    hh_of[children] = order[pick]

    if layout.cell_ids is not None:  # gridded population: sample a cell, then its settlement
        cw = layout.cell_weight / layout.cell_weight.sum()
        cell = rng.choice(len(cw), size=n_hh, p=cw).astype(np.int32)
        city = layout.cell_settlement[cell]
    else:
        cell = np.full(n_hh, -1, dtype=np.int32)
        city = rng.choice(len(layout.city_ids), size=n_hh, p=layout.city_share)
    equity = rng.pareto(cfg.wealth_pareto_alpha, size=n_hh) + 1.0
    households = HouseholdTable(capacity=int(n_hh * 1.5))
    households.add(n_hh, city=city.astype(np.int32), region=layout.city_region[city], cell=cell, weight=weight,
                   equity_share=equity / equity.sum())
    persons.hh = hh_of[pidx - pidx[0]]
    recount_members(persons, households)
    return persons, households
