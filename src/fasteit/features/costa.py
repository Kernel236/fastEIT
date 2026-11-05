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
    n_steps = len(peep)
    collapse = np.full(n_steps, np.nan)
    overdistension = np.full(n_steps, np.nan)

    # Steps with a compliance (a step without a driving pressure is all NaN).
    has_compliance = ~np.isnan(pixel_compliance).all(axis=(1, 2))
    if has_compliance.sum() < MIN_STEPS:
        return pd.DataFrame({"collapse": collapse, "overdistension": overdistension})
    known_steps = np.flatnonzero(has_compliance)
    known_peep = peep[known_steps]
    compliance = pixel_compliance[known_steps]  # (n_known_steps, rows, cols)

    # Best compliance of each pixel over the steps, and the PEEP where it is reached.
    # A missing value (NaN) must never be the best: it counts as -infinity here.
    comparable = np.where(np.isnan(compliance), -np.inf, compliance)
    best = comparable.max(axis=0)  # (rows, cols)
    best_peep = known_peep[comparable.argmax(axis=0)]  # (rows, cols)

    # Pixels used: inside the lung mask, with a positive best compliance.
    used = lung_mask & (best > 0)
    if not used.any():
        return pd.DataFrame({"collapse": collapse, "overdistension": overdistension})
    best_used = best[used]  # (n_pixels,)
    best_peep_used = best_peep[used]  # (n_pixels,)
    total_weight = best_used.sum()

    for row, step in enumerate(known_steps):
        # Loss of each pixel at this step, as % of its best; at most 100: a pixel that
        # is no more ventilated (its TIV around zero, with noise) lost all of it.
        loss = (best_used - compliance[row][used]) / best_used * 100
        loss = np.minimum(loss, 100)
        weighted_loss = loss * best_used
        # Below the best PEEP of a pixel its loss is collapse, above it overdistension.
        below_best = peep[step] < best_peep_used
        above_best = peep[step] > best_peep_used
        collapse[step] = np.nansum(weighted_loss[below_best]) / total_weight
        overdistension[step] = np.nansum(weighted_loss[above_best]) / total_weight
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

    Args:
        peep: PEEP of each step (cmH2O), in any order.
        collapse: Collapse of each step (%).
        overdistension: Overdistension of each step (%).

    Returns:
        The crossing PEEP; NaN if the curves do not cross.
    """
    # Sort the steps by PEEP, going up, and leave out those without values.
    order = np.argsort(peep, kind="stable")
    peep = np.asarray(peep, dtype=float)[order]
    collapse = np.asarray(collapse, dtype=float)[order]
    overdistension = np.asarray(overdistension, dtype=float)[order]
    difference = collapse - overdistension
    known = ~np.isnan(difference)
    peep, difference = peep[known], difference[known]

    # Steps where the curves are apart, and each pair of them in a row: a crossing is a
    # change from collapse above (positive) to overdistension above (negative).
    apart = np.flatnonzero(difference != 0)
    for lower, upper in zip(apart[:-1], apart[1:], strict=True):
        if difference[lower] > 0 > difference[upper]:
            if upper - lower > 1:
                # A step between them where the two are equal: that is the crossing.
                return float(peep[lower + 1])
            # Linear interpolation between the two steps.
            fraction = difference[lower] / (difference[lower] - difference[upper])
            return float(peep[lower] + fraction * (peep[upper] - peep[lower]))
    return np.nan
