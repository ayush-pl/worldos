"""Real-country geography from Natural Earth on the H3 grid.

* Country and admin-1 regions: polygons -> H3 cells (cell centroid containment, so regions are
  disjoint); a region too small for any cell gets the cell containing its representative point.
* Settlements: Natural Earth populated places (metropolitan ``pop_max``) form the urban part; each
  region also gets a "rest of region" settlement for its non-metropolitan population.
* Population allocation (an INFERRED downscaling, documented as crude): the urban share of the
  population (WDI) is split across listed cities by ``pop_max``; the remainder is split across
  regions in proportion to their area. Gridded population (WorldPop/GHSL) is the planned upgrade.
"""

from __future__ import annotations

import h3
import numpy as np
from shapely.geometry import mapping, shape

from worldos.geo.hierarchy import GeoEntity, GeoHierarchy, GeoLevel
from worldos.geo.synthetic import SettlementLayout


def _cells(geom, res: int) -> list[str]:
    try:
        cells = list(h3.geo_to_cells(mapping(geom), res))
    except Exception:  # degenerate geometry
        cells = []
    if not cells:
        pt = geom.representative_point()
        cells = [h3.latlng_to_cell(pt.y, pt.x, res)]
    return cells


def choose_resolution(area_km2: float, max_cells: int = 12_000) -> int:
    for res in range(6, 1, -1):
        if area_km2 / h3.average_hexagon_area(res, "km^2") <= max_cells:
            return res
    return 2


def build_country_geography(iso3: str, geojson: dict, urban_share: float, max_cities: int = 40,
                            ) -> tuple[GeoHierarchy, SettlementLayout, dict]:
    feats = geojson["features"]
    country_f = next(f for f in feats if f["properties"]["level"] == "country")
    country_geom = shape(country_f["geometry"])
    approx_area = country_geom.area * 111.0 * 111.0 * np.cos(np.radians(country_geom.centroid.y))
    res = choose_resolution(approx_area)
    geo = GeoHierarchy()
    geo.add(GeoEntity("earth", GeoLevel.PLANET, "Earth", None, (0.0, 0.0)))

    region_feats = [f for f in feats if f["properties"]["level"] == "region"]
    region_cells, region_ids, region_names = [], [], []
    taken: set[str] = set()
    for i, f in enumerate(region_feats):
        cells = [c for c in _cells(shape(f["geometry"]), res) if c not in taken]
        if not cells:
            continue
        taken.update(cells)
        region_cells.append(cells)
        region_ids.append(f"{iso3.lower()}.r{i + 1}")
        region_names.append(f["properties"]["name"] or f"Region {i + 1}")
    if not region_cells:  # no admin-1 data: whole country is one region
        region_cells = [_cells(country_geom, res)]
        region_ids, region_names = [f"{iso3.lower()}.r1"], [country_f["properties"]["name"]]
    all_cells = sorted(set().union(*region_cells))
    c = country_geom.representative_point()
    geo.add(GeoEntity(iso3.lower(), GeoLevel.COUNTRY, country_f["properties"]["name"], "earth", (c.y, c.x),
                      frozenset(all_cells), res, attributes={"iso3": iso3}))
    cell_region = {}
    for rid, name, cells in zip(region_ids, region_names, region_cells):
        pts = np.array([h3.cell_to_latlng(x) for x in cells])
        geo.add(GeoEntity(rid, GeoLevel.REGION, name, iso3.lower(), (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                          frozenset(cells), res))
        for x in cells:
            cell_region[x] = region_ids.index(rid)

    places = sorted((f for f in feats if f["properties"]["level"] == "city"),
                    key=lambda f: -float(f["properties"]["pop_max"]))[:max_cities]
    city_ids, city_region, city_pop, city_cell = [], [], [], []
    for j, f in enumerate(places):
        lng, lat = f["geometry"]["coordinates"][:2]
        parent = h3.latlng_to_cell(lat, lng, res)
        if parent not in cell_region:  # coastal city outside the simplified cell cover: nearest cell
            pts = np.array([h3.cell_to_latlng(x) for x in all_cells])
            parent = all_cells[int(((pts - [lat, lng]) ** 2).sum(axis=1).argmin())]
        r = cell_region[parent]
        cid = f"{region_ids[r]}.c{j + 1}"
        cell7 = h3.latlng_to_cell(lat, lng, max(res + 2, 7))
        geo.add(GeoEntity(cid, GeoLevel.CITY, f["properties"]["name"], region_ids[r], (lat, lng), frozenset([cell7]),
                          h3.get_resolution(cell7), attributes={"pop_max": f["properties"]["pop_max"],
                                                                "source": "Natural Earth populated places"}))
        city_ids.append(cid)
        city_region.append(r)
        city_pop.append(float(f["properties"]["pop_max"]))
        city_cell.append(cell7)
    # "rest of region" settlements carry the non-metropolitan population, allocated by area
    areas = np.array([sum(h3.cell_area(x, "km^2") for x in cells) for cells in region_cells])
    rural_share = (1.0 - urban_share) * areas / areas.sum()
    pop = np.array(city_pop)
    urban = urban_share * pop / pop.sum() if len(pop) else np.zeros(0)
    if not len(pop):
        rural_share = areas / areas.sum()
    for r, (rid, cells) in enumerate(zip(region_ids, region_cells)):
        cid = f"{rid}.rest"
        rep = cells[len(cells) // 2]
        lat, lng = h3.cell_to_latlng(rep)
        geo.add(GeoEntity(cid, GeoLevel.CITY, f"{region_names[r]} (rest of region)", rid, (lat, lng),
                          frozenset([rep]), res, attributes={"aggregate": True}))
        city_ids.append(cid)
        city_region.append(r)
        city_cell.append(rep)
    share = np.concatenate([urban, rural_share])
    share = share / share.sum()
    layout = SettlementLayout(region_ids, city_ids, np.array(city_region, dtype=np.int32), share, city_cell)
    info = {"h3_resolution": res, "n_cells": len(all_cells), "n_regions": len(region_ids), "n_cities": len(places),
            "urban_share": urban_share}
    return geo, layout, info
