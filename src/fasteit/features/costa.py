"""Collapse and overdistension in a PEEP trial, from the compliance of each pixel.

Costa et al., Intensive Care Med 2009;35:1132-1137, DOI: 10.1007/s00134-009-1447-y;
formulas as reported by Shono and Kotani, J Intensive Care 2019;7:4, DOI:
10.1186/s40560-019-0358-4 (notebooks/07_clinical_indices.ipynb, section 4):

- compliance of a pixel at a step = TIV of the pixel / driving pressure of the step;
- each pixel has its best compliance, at its best PEEP;
- at a step with a PEEP below the best PEEP of the pixel, its loss of compliance as %
  of its best is collapse; above, overdistension;
- collapse (overdistension) of a step = mean of the pixel %, weighted by the best
  compliance of each pixel, over the pixels of the lung mask;
- the crossing of the two curves is the PEEP with the best balance between them.

Compliance stays in EIT units: the method needs no tidal volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def collapse_and_overdistension(
    pixel_compliance: np.ndarray, peep: np.ndarray, lung_mask: np.ndarray
) -> pd.DataFrame:
    """Collapse and overdistension (%) of each step.

    Args:
        pixel_compliance: (n_steps, rows, cols) compliance of each pixel at each step
            (TIV / driving pressure); NaN where unknown, e.g. a step without a driving
            pressure.
        peep:      PEEP of each step.
        lung_mask: (rows, cols) bool, the pixels used.

    Returns:
        One row per step: ``collapse`` and ``overdistension`` (%); NaN for a step
        without compliance, and for every step if no pixel has a positive best
        compliance.
    """
    peep = np.asarray(peep, dtype=float)
    known = ~np.isnan(pixel_compliance).all(axis=(1, 2))
    collapse = np.full(len(peep), np.nan)
    overdistension = np.full(len(peep), np.nan)
    if not known.any():
        return pd.DataFrame({"collapse": collapse, "overdistension": overdistension})
    compliance = np.where(
        np.isnan(pixel_compliance[known]), -np.inf, pixel_compliance[known]
    )
    best = compliance.max(axis=0)
    used = lung_mask & (best > 0)
    if used.any():
        best = best[used]
        best_peep = peep[known][compliance.argmax(axis=0)[used]]
        loss = (best - compliance[:, used]) / best * 100  # % below the best, per step
        weight = best.sum()
        for i, k in enumerate(np.flatnonzero(known)):
            below, above = peep[k] < best_peep, peep[k] > best_peep
            collapse[k] = (loss[i, below] * best[below]).sum() / weight
            overdistension[k] = (loss[i, above] * best[above]).sum() / weight
    return pd.DataFrame({"collapse": collapse, "overdistension": overdistension})


def crossing_peep(
    peep: np.ndarray, collapse: np.ndarray, overdistension: np.ndarray
) -> float:
    """PEEP where the collapse and overdistension curves cross.

    Linear interpolation between the two steps where collapse - overdistension
    changes sign, going up in PEEP; if the curves cross more than once, the lowest
    crossing. A step on the crossing gives its PEEP.

    Returns:
        The crossing PEEP; NaN if the curves do not cross.
    """
    order = np.argsort(peep)
    peep = np.asarray(peep, dtype=float)[order]
    diff = (np.asarray(collapse) - np.asarray(overdistension))[order]
    keep = ~np.isnan(diff)
    peep, diff = peep[keep], diff[keep]
    for i in range(len(diff) - 1):
        if diff[i] >= 0 >= diff[i + 1] and diff[i] != diff[i + 1]:
            return float(
                peep[i] + (peep[i + 1] - peep[i]) * diff[i] / (diff[i] - diff[i + 1])
            )
    return np.nan
