from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_DATA = REPO_ROOT / "data" / "real"
DEFAULT_COUNTRIES = ["IND", "USA", "DEU", "JPN", "BRA", "NGA"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(folder: Path, source: str, license: str, entries: list[dict], notes: str = "") -> Path:
    for e in entries:
        e["sha256"] = sha256(folder / e["file"])
    manifest = {
        "source": source,
        "license": license,
        "retrieved_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
        "files": entries,
    }
    out = folder / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def verify_manifest(folder: Path) -> list[str]:
    """Return a list of files whose checksum no longer matches the manifest."""
    manifest = json.loads((folder / "manifest.json").read_text())
    return [e["file"] for e in manifest["files"] if sha256(folder / e["file"]) != e["sha256"]]
