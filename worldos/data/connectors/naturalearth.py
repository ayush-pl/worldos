"""Natural Earth connector (public domain): country outline, admin-1 regions, populated places.

Geometries are simplified (tolerance ~0.02 degrees, about 2 km) before storage; that is ample for
H3 resolutions 3-5 used by the geographic engine.
Citation: Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from shapely.geometry import mapping, shape

from worldos.data.connectors.common import DEFAULT_COUNTRIES, REAL_DATA, write_manifest

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
LICENSE = "Public domain (Natural Earth)"
TOLERANCE = 0.02


def _load(name: str) -> dict:
    with urllib.request.urlopen(BASE + name, timeout=600) as r:
        return json.loads(r.read())


def _iso(props: dict) -> str:
    for key in ("ADM0_A3", "adm0_a3", "ISO_A3", "iso_a3"):
        v = props.get(key)
        if v and v != "-99":
            return v
    return ""


def fetch_naturalearth(countries: list[str] | None = None, out_dir: Path | None = None) -> Path:
    countries = countries or DEFAULT_COUNTRIES
    out = out_dir or REAL_DATA / "naturalearth"
    out.mkdir(parents=True, exist_ok=True)
    adm0 = _load("ne_10m_admin_0_countries.geojson")
    adm1 = _load("ne_10m_admin_1_states_provinces.geojson")
    places = _load("ne_10m_populated_places_simple.geojson")
    entries = []
    for iso in countries:
        feats = []
        for f in adm0["features"]:
            if _iso(f["properties"]) == iso:
                geom = shape(f["geometry"]).simplify(TOLERANCE, preserve_topology=True)
                feats.append({"type": "Feature", "properties": {"level": "country", "iso3": iso,
                              "name": f["properties"].get("NAME") or f["properties"].get("ADMIN")},
                              "geometry": mapping(geom)})
        for f in adm1["features"]:
            p = f["properties"]
            if p.get("adm0_a3") == iso:
                geom = shape(f["geometry"]).simplify(TOLERANCE, preserve_topology=True)
                feats.append({"type": "Feature", "properties": {"level": "region", "iso3": iso, "name": p.get("name"),
                              "code": p.get("iso_3166_2") or p.get("adm1_code")}, "geometry": mapping(geom)})
        for f in places["features"]:
            p = f["properties"]
            if p.get("adm0_a3") == iso and (p.get("pop_max") or 0) > 0:
                feats.append({"type": "Feature", "properties": {"level": "city", "iso3": iso, "name": p.get("name"),
                              "region": p.get("adm1name"), "pop_max": p.get("pop_max"),
                              "capital": bool(p.get("adm0cap"))}, "geometry": f["geometry"]})
        if not any(ft["properties"]["level"] == "country" for ft in feats):
            raise ValueError(f"country {iso} not found in Natural Earth admin-0")
        name = f"{iso}.geojson"
        (out / name).write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
        entries.append({"file": name, "url": BASE, "rows": len(feats),
                        "filter": f"{iso}; 10m admin-0/admin-1 simplified {TOLERANCE} deg; populated places"})
    write_manifest(out, "Natural Earth 10m cultural vectors (nvkelso/natural-earth-vector)", LICENSE, entries)
    return out
