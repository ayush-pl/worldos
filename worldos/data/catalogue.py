"""Load and validate the dataset catalogue (worldos/data/catalogue.toml)."""

from __future__ import annotations

import tomllib
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    provider: str
    url: HttpUrl
    domains: list[str]
    spatial: str
    temporal: str
    license: str
    license_status: Literal["stated", "verify"]
    planned_use: str


def load_catalogue() -> list[DatasetEntry]:
    text = resources.files("worldos.data").joinpath("catalogue.toml").read_text(encoding="utf-8")
    entries = [DatasetEntry.model_validate(d) for d in tomllib.loads(text)["dataset"]]
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dataset ids in catalogue")
    return entries


def ingestion_allowed(entry: DatasetEntry, licence_confirmed_by: str | None = None) -> bool:
    """Ingestion of a dataset whose licence needs verification requires a named human confirmation."""
    return entry.license_status == "stated" or bool(licence_confirmed_by)
