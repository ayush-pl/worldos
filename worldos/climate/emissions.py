"""Territorial CO2 emissions accounting for a country economy (Kaya-style intensity model).

E_t = I_t * Y_t, with Y_t real output and I_t the CO2 intensity of output, which declines at the
configured decarbonisation rate (I_t = I_{t-1} * exp(-d/12) per month). For real countries I_0 and d
are INFERRED from Global Carbon Budget fossil CO2 and World Bank constant-price GDP. This is accounting,
not an energy-system model: it does not represent fuels, prices or abatement costs (Phase 3 energy engine).
"""

from __future__ import annotations

import numpy as np

from worldos.core.engine import Engine


class EmissionsEngine(Engine):
    name = "emissions"
    order = 55  # after production

    def step(self, world) -> None:
        c = world.config.climate
        if world.emissions_intensity is None:
            world.emissions_intensity = c.emissions_intensity
        else:
            world.emissions_intensity *= float(np.exp(-c.decarbonisation_rate / 12))
        world.flows.co2_emissions_t = world.emissions_intensity * world.flows.units_produced
