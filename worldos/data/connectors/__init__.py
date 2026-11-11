"""Connectors that fetch real data from catalogued sources into small, provenance-stamped extracts.

Extracts are written under ``data/real/<source>/`` together with a ``manifest.json`` recording the
source URL, retrieval timestamp, licence, filters applied and a SHA-256 of every output file, so
any downstream number can be traced back to its origin. Tests and calibration use the committed
extracts and never need network access.
"""
