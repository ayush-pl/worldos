"""Hierarchical geographic entities indexed on the H3 discrete global grid.

Every place has a stable id, a level in the geographic hierarchy, a parent, a
centroid, and a set of H3 cells (at a declared resolution) that defines its
spatial extent. H3 (Uber, Apache-2.0) is a hierarchical hexagonal DGGS that
supports moving between resolutions (``cell_to_parent`` / ``cell_to_children``),
which is how WORLD-OS changes geographic resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable

import h3


class GeoLevel(IntEnum):
    PLANET = 0
    CONTINENT = 1
    COUNTRY = 2
    REGION = 3  # state / province
    DISTRICT = 4  # county / district
    CITY = 5  # city / town / village
    NEIGHBOURHOOD = 6
    CELL = 7  # grid cell / location


@dataclass
class GeoEntity:
    id: str
    level: GeoLevel
    name: str
    parent: str | None
    centroid: tuple[float, float]
    cells: frozenset[str] = frozenset()
    resolution: int | None = None
    synthetic: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lat, lng = self.centroid
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError(f"invalid centroid {self.centroid} for {self.id}")
        if self.cells and self.resolution is None:
            raise ValueError("cells require a resolution")
        for c in self.cells:
            if h3.get_resolution(c) != self.resolution:
                raise ValueError(f"cell {c} not at resolution {self.resolution}")

    def area_km2(self) -> float:
        return float(sum(h3.cell_area(c, "km^2") for c in self.cells))

    def cells_at(self, resolution: int) -> set[str]:
        """Re-express the extent at another H3 resolution (coarser: parents, finer: children)."""
        if self.resolution is None:
            return set()
        if resolution == self.resolution:
            return set(self.cells)
        if resolution < self.resolution:
            return {h3.cell_to_parent(c, resolution) for c in self.cells}
        out: set[str] = set()
        for c in self.cells:
            out.update(h3.cell_to_children(c, resolution))
        return out


class GeoHierarchy:
    def __init__(self) -> None:
        self._entities: dict[str, GeoEntity] = {}
        self._children: dict[str, list[str]] = {}

    def add(self, entity: GeoEntity) -> GeoEntity:
        if entity.id in self._entities:
            raise ValueError(f"duplicate geo id {entity.id}")
        if entity.parent is not None:
            parent = self._entities.get(entity.parent)
            if parent is None:
                raise ValueError(f"unknown parent {entity.parent} for {entity.id}")
            if entity.level <= parent.level:
                raise ValueError(f"{entity.id} level {entity.level.name} must be below parent {parent.level.name}")
        self._entities[entity.id] = entity
        self._children.setdefault(entity.id, [])
        if entity.parent is not None:
            self._children[entity.parent].append(entity.id)
        return entity

    def __getitem__(self, geo_id: str) -> GeoEntity:
        return self._entities[geo_id]

    def __contains__(self, geo_id: str) -> bool:
        return geo_id in self._entities

    def __len__(self) -> int:
        return len(self._entities)

    def children(self, geo_id: str) -> list[GeoEntity]:
        return [self._entities[c] for c in self._children.get(geo_id, [])]

    def ancestors(self, geo_id: str) -> list[GeoEntity]:
        out = []
        cur = self._entities[geo_id].parent
        while cur is not None:
            out.append(self._entities[cur])
            cur = self._entities[cur].parent
        return out

    def at_level(self, level: GeoLevel) -> list[GeoEntity]:
        return [e for e in self._entities.values() if e.level == level]

    def locate(self, lat: float, lng: float, level: GeoLevel) -> GeoEntity | None:
        """Return the entity at ``level`` whose cells contain the point, if any."""
        for e in self.at_level(level):
            if e.resolution is not None and h3.latlng_to_cell(lat, lng, e.resolution) in e.cells:
                return e
        return None

    def entities(self) -> Iterable[GeoEntity]:
        return self._entities.values()
