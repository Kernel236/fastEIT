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

Compliance stays in EIT units: the method needs no tidal volume. An error common to
all the pixels of a step (its driving pressure, the share of the tidal volume seen by
the slice) shifts them together and appears as collapse or overdistension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fewest steps with a compliance for the method to compare them.
MIN_STEPS = 2


def collapse_and_overdistension(
    pixel_compliance: np.ndarray, peep: np.ndarray, lung_mask: np.ndarray
) -> pd.DataFrame:
    """Collapse and overdistension (%) of each step (Costa 2009, formulas of Shono and
    Kotani 2019, see the module docstring).

    collapse(step) = sum(loss_pixel(step) x best_pixel) / sum(best_pixel) over the
    pixels whose best PEEP is above the PEEP of the step, with loss_pixel = (best_pixel
    - compliance_pixel(step)) / best_pixel x 100; overdistension the same over the
    pixels whose best PEEP is below it. The loss of a pixel is at most 100 %.

    Args:
        pixel_compliance: (n_steps, rows, cols) compliance of each pixel at each step
            (TIV / driving pressure); NaN where unknown, e.g. a step without a driving
            pressure.
        peep:      PEEP of each step.
        lung_mask: (rows, cols) bool, the pixels used.

    Returns:
        One row per step: ``collapse`` and ``overdistension`` (%); NaN for a step
        without compliance, and for every step if fewer than 2 steps have one or no
        pixel has a positive best compliance.
    """
    peep = np.asarray(peep, dtype=float)
    known = ~np.isnan(pixel_compliance).all(axis=(1, 2))
    collapse = np.full(len(peep), np.nan)
    overdistension = np.full(len(peep), np.nan)
    if known.sum() < MIN_STEPS:
        return pd.DataFrame({"collapse": collapse, "overdistension": overdistension})
    compliance = pixel_compliance[known]
    filled = np.where(np.isnan(compliance), -np.inf, compliance)
    best = filled.max(axis=0)
    used = lung_mask & (best > 0)
    if used.any():
        best = best[used]
        best_peep = peep[known][filled.argmax(axis=0)[used]]
        # % below the best, per step; at most 100: a pixel that is no more ventilated
        # (its TIV around zero, with noise) has lost all of its compliance.
        loss = np.minimum((best - compliance[:, used]) / best * 100, 100)
        weight = best.sum()
        for i, k in enumerate(np.flatnonzero(known)):
            below, above = peep[k] < best_peep, peep[k] > best_peep
            collapse[k] = np.nansum(loss[i, below] * best[below]) / weight
            overdistension[k] = np.nansum(loss[i, above] * best[above]) / weight
    return pd.DataFrame({"collapse": collapse, "overdistension": overdistension})


def crossing_peep(
    peep: np.ndarray, collapse: np.ndarray, overdistension: np.ndarray
) -> float:
    """PEEP where the collapse and overdistension curves cross (Costa 2009).

    Going up in PEEP, collapse - overdistension must change sign, from positive to
    negative: linear interpolation between the two steps, or the PEEP of a step where
    the two are equal between them. Curves that only touch (the difference goes back
    to positive, or stays at 0 up to the highest PEEP) do not cross. If they cross more
    than once, the lowest crossing.

    Returns:
        The crossing PEEP; NaN if the curves do not cross.
    """
    order = np.argsort(peep, kind="stable")
    peep = np.asarray(peep, dtype=float)[order]
    diff = (
        np.asarray(collapse, dtype=float) - np.asarray(overdistension, dtype=float)
    )[order]
    keep = ~np.isnan(diff)
    peep, diff = peep[keep], diff[keep]
    apart = np.flatnonzero(diff != 0)
    for a, b in zip(apart[:-1], apart[1:], strict=True):
        if diff[a] > 0 > diff[b]:
            if b - a > 1:  # a step with no difference between them
                return float(peep[a + 1])
            return float(peep[a] + (peep[b] - peep[a]) * diff[a] / (diff[a] - diff[b]))
    return np.nan
