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

By region: the same formula on the pixels of the region only, its denominator included,
so the value is a % of the region ("30 % of the dorsal lung collapsed"); the best of each
pixel is its own, over all the steps. Published by right and left lung (Mlcek et al., BMC
Pulm Med 2021;21, DOI: 10.1186/s12890-021-01501-x, as % of the whole lung) and by
ventral and dorsal half of the image (Jonkman et al., Am J Respir Crit Care Med
2023;208:25-38, DOI: 10.1164/rccm.202212-2300OC). The weight of a region,
``costa_weight`` = its best compliance / that of the lung, gives the % of the whole lung
(% of the region x weight), and on a partition of the lung the weighted mean of the
regions is the value of the lung.
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

    known_steps = _steps_with_compliance(pixel_compliance)
    if known_steps is None:
        return pd.DataFrame({"collapse": collapse, "overdistension": overdistension})
    known_peep = peep[known_steps]
    compliance = pixel_compliance[known_steps]  # (n_known_steps, rows, cols)

    # Best compliance of each pixel over the steps, and the PEEP where it is reached.
    best, best_step = _best_compliance(compliance)
    best_peep = known_peep[best_step]  # (rows, cols)

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


def costa_weight(
    pixel_compliance: np.ndarray, region: np.ndarray, lung_mask: np.ndarray
) -> float:
    """Weight of a region in Costa: its best compliance / that of the lung (0-1).

    The sums of the best compliance of the pixels used by ``collapse_and_overdistension``
    (positive best), in the region and in the lung mask. Collapse (overdistension) of the
    region x its weight = its part of the value of the lung; a small weight marks a
    region small or little ventilated, where its % is unstable.

    Args:
        pixel_compliance: (n_steps, rows, cols), as in ``collapse_and_overdistension``.
        region:    (rows, cols) bool, inside the lung mask.
        lung_mask: (rows, cols) bool.

    Returns:
        The weight; 0 for a region without pixels used; NaN with fewer than 2 steps
        with a compliance or no pixel used in the lung.
    """
    known_steps = _steps_with_compliance(pixel_compliance)
    if known_steps is None:
        return np.nan
    best, _ = _best_compliance(pixel_compliance[known_steps])
    lung = best[lung_mask & (best > 0)].sum()
    if lung == 0:
        return np.nan
    return float(best[region & lung_mask & (best > 0)].sum() / lung)


def _steps_with_compliance(pixel_compliance: np.ndarray) -> np.ndarray | None:
    """Steps with a compliance (a step without a driving pressure is all NaN); None if
    fewer than ``MIN_STEPS``."""
    has_compliance = ~np.isnan(pixel_compliance).all(axis=(1, 2))
    if has_compliance.sum() < MIN_STEPS:
        return None
    return np.flatnonzero(has_compliance)


def _best_compliance(compliance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best compliance of each pixel over the steps, and the index of its step.

    A missing value (NaN) must never be the best: it counts as -infinity here.
    """
    comparable = np.where(np.isnan(compliance), -np.inf, compliance)
    return comparable.max(axis=0), comparable.argmax(axis=0)


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
