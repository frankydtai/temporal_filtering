"""Shared plotting helpers (no target-specific logic)."""

from __future__ import annotations

import numpy as np


def nice_ylim(*curves, margin=1.25, step=5.0, floor=5.0, min_pad=3.0):
    """Symmetric y-limits that comfortably contain all provided curves."""
    vals = [np.asarray(c).ravel() for c in curves if c is not None]
    if not vals:
        return -floor, floor
    peak = float(np.max(np.abs(np.concatenate(vals))))
    ymax = max(peak * margin, peak + min_pad, floor)
    ymax = float(np.ceil(ymax / step) * step)
    return -ymax, ymax

