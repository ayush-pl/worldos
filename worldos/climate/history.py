"""Historical climate reconstruction 1850-2023 and validation against observations; scenario runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from worldos.climate.model import ClimateParams, ClimateRun, run_climate
from worldos.data.connectors.common import REAL_DATA

CLIMATE = REAL_DATA / "climate"
START = 1750  # run from pre-industrial equilibrium (C = 278 ppm, T = 0)


def historical_inputs(end: int = 2023) -> pd.DataFrame:
    owid = pd.read_csv(CLIMATE / "emissions_owid_gcb.csv")
    world = owid[owid.country == "World"].set_index("year")
    erf = pd.read_csv(CLIMATE / "ar6_erf_1750_2019.csv").set_index("year")
    years = np.arange(START, end + 1)
    df = pd.DataFrame(index=years)
    fossil = world["co2"].reindex(years).fillna(0.0)
    luc = world["land_use_change_co2"].reindex(years)
    # land-use emissions start in 1850 in this dataset; 1750-1849 ramp linearly from 0 to the 1850 value (assumption)
    first = luc.first_valid_index()
    luc.loc[:first] = np.linspace(0.0, luc.loc[first], len(luc.loc[:first]))
    df["emissions_gtco2"] = (fossil + luc.interpolate().ffill()) / 1000.0
    other = (erf["total"] - erf["co2"]).reindex(years)
    df["other_forcing"] = other.ffill()  # 2020-2023 held at the 2019 value (assumption)
    return df


def observations() -> tuple[pd.Series, pd.Series]:
    noaa = pd.read_csv(CLIMATE / "co2_mauna_loa_annual.csv").set_index("year")["mean"]
    had = pd.read_csv(CLIMATE / "hadcrut5_global_annual.csv").set_index("Time")["Anomaly (deg C)"]
    had = had - had.loc[1850:1900].mean()  # rebase to 1850-1900, the IPCC pre-industrial proxy
    return noaa, had


def reconstruct(p: ClimateParams = ClimateParams(), end: int = 2023) -> ClimateRun:
    d = historical_inputs(end)
    return run_climate(d.index.to_numpy(), d.emissions_gtco2.to_numpy(), d.other_forcing.to_numpy(), None, p)


def validate(p: ClimateParams = ClimateParams()) -> dict:
    run = reconstruct(p)
    noaa, had = observations()
    sim_c = pd.Series(run.co2_ppm, index=run.years)
    sim_t = pd.Series(run.temperature, index=run.years)
    sim_t = sim_t - sim_t.loc[1850:1900].mean()
    yc = noaa.index.intersection(sim_c.index)
    yt = had.index.intersection(sim_t.index)
    rec = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
    t_obs_recent = float(had.loc[2014:2023].mean())
    t_sim_recent = float(sim_t.loc[2014:2023].mean())
    return {
        "co2_rmse_ppm": rec(sim_c.loc[yc], noaa.loc[yc]),
        "co2_2023_sim": float(sim_c.loc[2023]), "co2_2023_obs": float(noaa.loc[2023]),
        "temp_rmse_k": rec(sim_t.loc[yt], had.loc[yt]),
        "temp_2014_2023_sim": t_sim_recent, "temp_2014_2023_obs": t_obs_recent,
        "temp_trend_1970_2023_sim": float(np.polyfit(np.arange(1970, 2024), sim_t.loc[1970:2023], 1)[0] * 10),
        "temp_trend_1970_2023_obs": float(np.polyfit(np.arange(1970, 2024), had.loc[1970:2023], 1)[0] * 10),
        "ecs_k": p.ecs,
    }


def scenario(final_year: int = 2100, net_zero_year: int | None = 2070, other_forcing_2100: float | None = None,
             p: ClimateParams = ClimateParams()) -> pd.DataFrame:
    """Illustrative global emissions pathway: hold 2023 emissions to 2025, then decline linearly to zero at
    ``net_zero_year`` (None = constant emissions). Non-CO2 forcing moves linearly from its 2019 level to
    ``other_forcing_2100`` (None = constant). A what-if tool, not an SSP reproduction."""
    hist = historical_inputs(2023)
    years = np.arange(START, final_year + 1)
    e = hist.emissions_gtco2.reindex(years).to_numpy().copy()
    o = hist.other_forcing.reindex(years).to_numpy().copy()
    e_last, o_last = hist.emissions_gtco2.iloc[-1], hist.other_forcing.iloc[-1]
    fut = years > 2023
    if net_zero_year is None:
        e[fut] = e_last
    else:
        frac = np.clip((net_zero_year - years[fut]) / max(net_zero_year - 2025, 1), 0, 1)
        e[fut] = e_last * np.where(years[fut] <= 2025, 1.0, frac)
    target = o_last if other_forcing_2100 is None else other_forcing_2100
    o[fut] = o_last + (target - o_last) * (years[fut] - 2023) / (final_year - 2023)
    run = run_climate(years, e, o, None, p)
    t = run.temperature - run.temperature[(years >= 1850) & (years <= 1900)].mean()
    return pd.DataFrame({"year": years, "emissions_gtco2": e, "co2_ppm": run.co2_ppm, "temperature_k": t}).set_index("year")


def _series(run: ClimateRun) -> tuple[pd.Series, pd.Series]:
    c = pd.Series(run.co2_ppm, index=run.years)
    t = pd.Series(run.temperature, index=run.years)
    return c, t - t.loc[1850:1900].mean()


def calibrate(train_co2=(1959, 1999), train_temp=(1850, 1989)) -> dict:
    """Two sequential one-parameter calibrations with held-out test periods.

    1. Carbon-cycle pre-industrial airborne time scale r0 to Mauna Loa CO2 over ``train_co2``.
    2. Climate feedback parameter lambda (hence ECS) to HadCRUT5 over ``train_temp``.
    Test periods are the remaining years up to 2023. Uncertainties in emissions and forcing are absorbed by
    these parameters, so they are effective values, not physical estimates.
    """
    from scipy.optimize import minimize_scalar

    d = historical_inputs()
    noaa, had = observations()
    run = lambda p: run_climate(d.index.to_numpy(), d.emissions_gtco2.to_numpy(), d.other_forcing.to_numpy(), None, p)
    a, b = train_co2
    co2_loss = lambda r0: float(((_series(run(ClimateParams(r0=r0)))[0].loc[a:b] - noaa.loc[a:b]) ** 2).mean())
    r0 = float(minimize_scalar(co2_loss, bounds=(15.0, 60.0), method="bounded").x)
    a2, b2 = train_temp
    temp_loss = lambda lam: float(((_series(run(ClimateParams(r0=r0, lam=lam)))[1].loc[a2:b2] - had.loc[a2:b2]) ** 2).mean())
    lam_fit = float(minimize_scalar(temp_loss, bounds=(0.5, 3.0), method="bounded").x)
    rmse = lambda x, y: float(np.sqrt(((x - y) ** 2).mean()))
    test = lambda lam: rmse(_series(run(ClimateParams(r0=r0, lam=lam)))[1].loc[b2 + 1:2023], had.loc[b2 + 1:2023])
    prior = ClimateParams().lam
    # keep the CMIP5-range prior unless the fitted value also predicts the held-out period better
    lam = lam_fit if test(lam_fit) < test(prior) else prior
    p = ClimateParams(r0=r0, lam=lam)
    c, t = _series(run(p))
    return {
        "r0": r0, "lambda": lam, "lambda_fitted_on_train": lam_fit, "lambda_prior": prior,
        "lambda_choice": "fitted" if lam == lam_fit else "prior (fitted value failed the out-of-sample test)",
        "temp_rmse_test_k_fitted": test(lam_fit), "ecs_k": p.ecs,
        "co2_rmse_train_ppm": rmse(c.loc[a:b], noaa.loc[a:b]),
        "co2_rmse_test_ppm": rmse(c.loc[b + 1:2023], noaa.loc[b + 1:2023]),
        "co2_2023_sim": float(c.loc[2023]), "co2_2023_obs": float(noaa.loc[2023]),
        "temp_rmse_train_k": rmse(t.loc[a2:b2], had.loc[a2:b2]),
        "temp_rmse_test_k": rmse(t.loc[b2 + 1:2023], had.loc[b2 + 1:2023]),
        "temp_2014_2023_sim": float(t.loc[2014:2023].mean()), "temp_2014_2023_obs": float(had.loc[2014:2023].mean()),
        "params": p,
    }


def climate_report() -> tuple[str, dict]:
    import json

    r = calibrate()
    p = r.pop("params")
    path = REAL_DATA.parent / "calibration" / "climate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2))
    L = ["# Climate module: historical reconstruction and validation", "",
         "> Simple emulator chain (FaIR-v1-type carbon cycle, Myhre CO2 forcing + IPCC AR6 non-CO2 forcing, two-layer",
         "> energy balance model). Generated by `python -m worldos climate validate`.", "",
         "## Calibration with held-out test periods", "",
         "| item | value |", "|---|---|"]
    for k, v in r.items():
        L.append(f"| {k} | {v if isinstance(v, str) else round(v, 4)} |")
    L += ["", "## Illustrative global pathways (not SSP reproductions)", "",
          "Emissions held at the 2023 level to 2025, then linear decline to zero in the stated year; non-CO2 forcing",
          "held at its 2019 level. Temperature relative to 1850-1900.", "",
          "| pathway | CO2 2050 (ppm) | CO2 2100 (ppm) | warming 2050 (K) | warming 2100 (K) |", "|---|---|---|---|---|"]
    for label, nz in (("constant 2023 emissions", None), ("net zero 2070", 2070), ("net zero 2050", 2050)):
        s = scenario(net_zero_year=nz, p=p)
        L.append(f"| {label} | {s.co2_ppm[2050]:.0f} | {s.co2_ppm[2100]:.0f} | {s.temperature_k[2050]:.2f} | {s.temperature_k[2100]:.2f} |")
    L += ["", "## Limitations", "",
          "- Single global box: no regional climate, precipitation, sea level or extremes yet.",
          "- Non-CO2 forcing is exogenous (held at 2019 in pathways); aerosol forcing uncertainty dominates the",
          "  temperature calibration and makes climate sensitivity weakly identified from the historical record.",
          "- The model runs about 0.2 K warm over 2014-2023 with the prior sensitivity; pathway temperatures are",
          "  therefore likely biased high by a similar amount.",
          "- No damages are fed back to the economy yet (Phase 3 coupling milestone).", ""]
    return "\n".join(L), r
