import h3
import numpy as np

from worldos.config import GeoConfig
from worldos.geo.hierarchy import GeoLevel
from worldos.geo.synthetic import build_synthetic_geography


def test_regions_partition_country_cells():
    geo, layout = build_synthetic_geography(GeoConfig(), np.random.default_rng(0))
    country = geo["syn"]
    regions = geo.children("syn")
    assert len(regions) == GeoConfig().n_regions
    union = set().union(*(r.cells for r in regions))
    assert union == set(country.cells)
    assert sum(len(r.cells) for r in regions) == len(country.cells)  # disjoint


def test_cities_inside_their_region_and_zipf_shares():
    geo, layout = build_synthetic_geography(GeoConfig(), np.random.default_rng(1))
    assert abs(layout.city_share.sum() - 1) < 1e-12
    assert np.all(np.diff(layout.city_share) < 0)  # rank-size: decreasing
    for cid in layout.city_ids:
        city = geo[cid]
        region = geo[city.parent]
        parent_cell = h3.cell_to_parent(next(iter(city.cells)), region.resolution)
        assert parent_cell in region.cells
        assert [a.level for a in geo.ancestors(cid)] == [GeoLevel.REGION, GeoLevel.COUNTRY, GeoLevel.PLANET]


def test_resolution_changes_are_consistent():
    geo, _ = build_synthetic_geography(GeoConfig(), np.random.default_rng(2))
    country = geo["syn"]
    coarse = country.cells_at(country.resolution - 1)
    assert all(h3.get_resolution(c) == country.resolution - 1 for c in coarse)
    fine = geo.children("syn")[0].cells_at(country.resolution + 1)
    assert len(fine) == 7 * len(geo.children("syn")[0].cells) - sum(
        1 for c in geo.children("syn")[0].cells if h3.is_pentagon(c))
