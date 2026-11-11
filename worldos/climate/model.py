"""A simple, established climate emulator chain, annual time step.

1. Carbon cycle: the state-dependent impulse-response model of FaIR v1 (Millar et al. 2017,
   Atmos. Chem. Phys. 17, 7213; Smith et al. 2018, GMD 11, 2273). Four reservoirs with the Joos et al.
   (2013) partition a = (0.2173, 0.2240, 0.2824, 0.2763) and time scales tau = (1e6, 394.4, 36.54,
   4.304) yr, all scaled by alpha so that the 100-year integrated impulse response equals
   iIRF100 = r0 + rC * C_acc + rT * T  (r0 = 35 yr, rC = 0.019 yr/GtC, rT = 4.165 yr/K),
   which represents weakening sinks as cumulative uptake and temperature rise.
   (The state-independent Joos IRF alone overestimated 2023 CO2 by ~50 ppm in our tests.)
2. CO2 forcing: F = 5.35 ln(C / C0) W m-2 (Myhre et al. 1998), C0 = 278 ppm.
   Non-CO2 forcing (other GHGs, aerosols, land use, solar, volcanic) is exogenous: by default the
   IPCC AR6 effective radiative forcing series minus its CO2 component.
3. Temperature: two-layer energy balance model (Held et al. 2010; Geoffroy et al. 2013):
       C  dT/dt  = F - lambda T - gamma (T - T_D)
       C_D dT_D/dt = gamma (T - T_D)
   Default parameters lie in the CMIP5 range reported by Geoffroy et al. (2013):
   lambda = 1.13 W m-2 K-1 (ECS = 3.71/lambda ~ 3.3 K), C = 8 and C_D = 100 W yr m-2 K-1, gamma = 0.7.

Units: emissions GtCO2/yr; 1 ppm CO2 = 2.124 GtC = 7.782 GtCO2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GTCO2_PER_PPM = 2.124 * 3.664
JOOS_A = np.array([0.2173, 0.2240, 0.2824, 0.2763])
JOOS_TAU = np.array([394.4, 36.54, 4.304])
FAIR_TAU = np.array([1.0e6, 394.4, 36.54, 4.304])
PPM_PER_GTC = 1 / 2.124


@dataclass(frozen=True)
class ClimateParams:
    c0_ppm: float = 278.0
    r0: float = 35.0
    rc: float = 0.019
    rt: float = 4.165
    forcing_2x: float = 3.71
    lam: float = 1.13
    heat_capacity: float = 8.0
    deep_heat_capacity: float = 100.0
    gamma: float = 0.7

    @property
    def ecs(self) -> float:
        return self.forcing_2x / self.lam


def irf(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    return JOOS_A[0] + (JOOS_A[1:, None] * np.exp(-t[None, :] / JOOS_TAU[:, None])).sum(axis=0)


def concentration(emissions_gtco2: np.ndarray, c_start_ppm: float, c0_ppm: float = 278.0) -> np.ndarray:
    """CO2 concentration at the END of each year from annual emissions after the start year.

    The excess over pre-industrial present at the start (c_start - c0) is assumed to persist (simplification).
    """
    e = np.asarray(emissions_gtco2, float) / GTCO2_PER_PPM
    n = len(e)
    kernel = irf(np.arange(n) + 0.5)
    return c_start_ppm + np.convolve(e, kernel)[:n]


def co2_forcing(c_ppm: np.ndarray, p: ClimateParams = ClimateParams()) -> np.ndarray:
    return 5.35 * np.log(np.asarray(c_ppm) / p.c0_ppm)


def temperature(forcing: np.ndarray, p: ClimateParams = ClimateParams(), t0: float = 0.0, td0: float = 0.0,
                substeps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Surface and deep-ocean temperature anomalies (K) at the end of each year (explicit sub-stepping)."""
    T, TD = t0, td0
    dt = 1.0 / substeps
    out_t, out_d = np.zeros(len(forcing)), np.zeros(len(forcing))
    for i, F in enumerate(forcing):
        for _ in range(substeps):
            dT = (F - p.lam * T - p.gamma * (T - TD)) / p.heat_capacity
            dTD = p.gamma * (T - TD) / p.deep_heat_capacity
            T, TD = T + dt * dT, TD + dt * dTD
        out_t[i], out_d[i] = T, TD
    return out_t, out_d


@dataclass
class ClimateRun:
    years: np.ndarray
    emissions: np.ndarray
    co2_ppm: np.ndarray
    forcing_co2: np.ndarray
    forcing_other: np.ndarray
    temperature: np.ndarray  # K relative to the start year


def _alpha(iirf100: float) -> float:
    """Solve sum_i a_i alpha tau_i (1 - exp(-100 / (alpha tau_i))) = iIRF100 for alpha (bisection in log space)."""
    g = lambda al: float((JOOS_A * al * FAIR_TAU * (1 - np.exp(-100.0 / (al * FAIR_TAU)))).sum()) - iirf100
    lo, hi = 1e-4, 1e2
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
    return float(np.sqrt(lo * hi))


def run_climate(years: np.ndarray, emissions_gtco2: np.ndarray, other_forcing: np.ndarray, c_start_ppm: float | None = None,
                p: ClimateParams = ClimateParams()) -> ClimateRun:
    """Coupled annual run: FaIR-type carbon cycle, CO2 + exogenous forcing, two-layer EBM. Starts in equilibrium."""
    n = len(years)
    e_gtc = np.asarray(emissions_gtco2, float) / 3.664
    R = np.zeros(4)
    if c_start_ppm is not None and c_start_ppm > p.c0_ppm:  # place any initial excess in the reservoirs
        R += JOOS_A * (c_start_ppm - p.c0_ppm) / PPM_PER_GTC
    cum_e, T, TD = float(R.sum()), 0.0, 0.0
    conc, f_co2, temp = np.zeros(n), np.zeros(n), np.zeros(n)
    dt = 1.0 / 12
    for i in range(n):
        c_acc = cum_e - R.sum()
        alpha = _alpha(min(p.r0 + p.rc * c_acc + p.rt * T, 97.0))
        decay = np.exp(-1.0 / (alpha * FAIR_TAU))
        R = R * decay + JOOS_A * e_gtc[i] * (1 + decay) / 2  # emissions spread through the year
        cum_e += e_gtc[i]
        conc[i] = p.c0_ppm + R.sum() * PPM_PER_GTC
        f_co2[i] = 5.35 * np.log(conc[i] / p.c0_ppm)
        F = f_co2[i] + other_forcing[i]
        for _ in range(12):
            dT = (F - p.lam * T - p.gamma * (T - TD)) / p.heat_capacity
            dTD = p.gamma * (T - TD) / p.deep_heat_capacity
            T, TD = T + dt * dT, TD + dt * dTD
        temp[i] = T
    return ClimateRun(np.asarray(years), np.asarray(emissions_gtco2), conc, f_co2, np.asarray(other_forcing), temp)
