"""Construction of the synthetic Alpha country "Synthland".

Real countries will be built from Natural Earth / national boundary data and
GHSL / WorldPop settlement grids (data catalogue ids ``natural_earth_admin``,
``ghsl``, ``worldpop``). Alpha uses a synthetic territory so that no output
can be mistaken for a statement about a real place:

* territory: H3 cells covering a configured polygon (default over open ocean);
* regions: a seeded Voronoi partition of those cells (nearest of k seeds);
* cities: placed on random cells, with population shares from the rank-size
  (Zipf) rule, an empirical regularity of city-size distributions (Zipf 1949;
  Gabaix 1999), used here only as a generative assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import h3
import numpy as np

from worldos.config import GeoConfig
from worldos.geo.hierarchy import GeoEntity, GeoHierarchy, GeoLevel


@dataclass
class SettlementLayout:
    """Array view of settlements used by vectorised engines."""

    region_ids: list[str]
    city_ids: list[str]
    city_region: np.ndarray  # int index into region_ids
    city_share: np.ndarray  # population share, sums to 1
    city_cell: list[str]  # H3 cell at city resolution


def _centroid(cells: list[str]) -> tuple[float, float]:
    pts = np.array([h3.cell_to_latlng(c) for c in cells])
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def build_synthetic_geography(cfg: GeoConfig, rng: np.random.Generator) -> tuple[GeoHierarchy, SettlementLayout]:
    geo = GeoHierarchy()
    geo.add(GeoEntity("earth", GeoLevel.PLANET, "Earth", None, (0.0, 0.0)))
    cells = sorted(h3.h3shape_to_cells(h3.LatLngPoly(list(cfg.polygon)), cfg.country_resolution))
    if len(cells) < cfg.n_regions:
        raise ValueError("polygon too small for the requested number of regions at this resolution")
    country = geo.add(
        GeoEntity("syn", GeoLevel.COUNTRY, cfg.country_name, "earth", _centroid(cells), frozenset(cells),
                  cfg.country_resolution, synthetic=True)
    )

    latlng = np.array([h3.cell_to_latlng(c) for c in cells])
    seeds = latlng[rng.choice(len(cells), size=cfg.n_regions, replace=False)]
    # Squared-degree distance is adequate for assigning cells to nearest seed near the equator.
    d = ((latlng[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    owner = d.argmin(axis=1)
    region_ids = []
    for r in range(cfg.n_regions):
        rcells = [c for c, o in zip(cells, owner) if o == r]
        if not rcells:  # seed cell always belongs to its own region, so this cannot happen
            raise RuntimeError("empty region")
        rid = f"syn.r{r + 1}"
        geo.add(GeoEntity(rid, GeoLevel.REGION, f"Region {r + 1}", country.id, _centroid(rcells),
                          frozenset(rcells), cfg.country_resolution, synthetic=True))
        region_ids.append(rid)

    ranks = np.arange(1, cfg.n_cities + 1, dtype=float)
    share = ranks ** (-cfg.zipf_exponent)
    share /= share.sum()
    city_cells_parent = rng.choice(len(cells), size=cfg.n_cities, replace=False)
    city_ids, city_region, city_cell = [], [], []
    for i, ci in enumerate(city_cells_parent):
        parent_cell = cells[ci]
        lat, lng = h3.cell_to_latlng(parent_cell)
        ccell = h3.latlng_to_cell(lat, lng, cfg.city_resolution)
        r = int(owner[ci])
        cid = f"{region_ids[r]}.c{i + 1}"
        geo.add(GeoEntity(cid, GeoLevel.CITY, f"City {i + 1}", region_ids[r], (lat, lng),
                          frozenset([ccell]), cfg.city_resolution, synthetic=True,
                          attributes={"population_share": float(share[i]), "rank": i + 1}))
        city_ids.append(cid)
        city_region.append(r)
        city_cell.append(ccell)
    layout = SettlementLayout(region_ids, city_ids, np.array(city_region, dtype=np.int32), share, city_cell)
    return geo, layout
