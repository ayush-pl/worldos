"""Distributional statistics with survey-style weights."""

from __future__ import annotations

import numpy as np


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if len(x) == 0:
        return float("nan")
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    return float(x[np.searchsorted(cw, q * cw[-1], side="left").clip(0, len(x) - 1)])


def gini(x: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted Gini coefficient (trapezoidal Lorenz curve). Negative values are allowed (wealth)."""
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    if len(x) == 0 or w.sum() <= 0:
        return float("nan")
    order = np.argsort(x)
    x, w = x[order], w[order]
    total = (x * w).sum()
    if total == 0:
        return 0.0
    cw = np.cumsum(w) / w.sum()
    cx = np.cumsum(x * w) / total
    prev_cw = np.r_[0.0, cw[:-1]]
    prev_cx = np.r_[0.0, cx[:-1]]
    area = ((cw - prev_cw) * (cx + prev_cx)).sum()
    return float(1.0 - area)
