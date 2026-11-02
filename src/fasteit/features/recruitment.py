"""Recruitment-to-inflation ratio (R/I) of a PEEP trial, from EIT.

R/I (Chen et al., Am J Respir Crit Care Med 2020;201:178-187, DOI:
10.1164/rccm.201902-0334OC): when PEEP falls from a high to a low level, the lung loses
more volume than its compliance at the low PEEP predicts; the excess is the volume that
was recruited at the high PEEP:

    predicted volume  = C_low x dPEEP
    recruited volume  = dEELV - C_low x dPEEP
    R/I               = (recruited volume / dPEEP) / C_low

With EIT the change of end-expiratory lung volume is the change of EELI converted with
the tidal volume / TIV of the low PEEP step (Wang et al., Med Sci Monit
2020;26:e922609, DOI: 10.12659/MSM.922609), and C_low = tidal volume / driving pressure
of the low step. The tidal volume cancels in R/I, which therefore needs only EIT and the
driving pressure:

    R/I = dEELI / (C_EIT,low x dPEEP) - 1,   with C_EIT,low = TIV_low / driving_low

and it applies to a region with its own EELI and EIT compliance. The original method
reads one expiration after the drop of PEEP; here each level is the equilibrium EELI of
a step, as in Wang 2020.
"""

from __future__ import annotations

import numpy as np


def ri_steps(
    peep: np.ndarray, high: float, low: float, min_drop: float
) -> tuple[int, int] | None:
    """The two steps used for R/I: the nearest to the high and to the low PEEP.

    The high step is the step nearest to ``high``; the low step is the step nearest to
    ``low`` among those at least ``min_drop`` below the high step (the first one if two
    are equally near).

    Args:
        peep:     PEEP of each step.
        high:     Target high PEEP (e.g. 15).
        low:      Target low PEEP (e.g. 5).
        min_drop: Smallest drop of PEEP between the two steps.

    Returns:
        Index of the high and of the low step, or None if no pair drops enough.
    """
    peep = np.asarray(peep, dtype=float)
    known = np.flatnonzero(~np.isnan(peep))
    if len(known) < 2:
        return None
    high_step = known[np.argmin(np.abs(peep[known] - high))]
    far_enough = known[peep[known] <= peep[high_step] - min_drop]
    if len(far_enough) == 0:
        return None
    low_step = far_enough[np.argmin(np.abs(peep[far_enough] - low))]
    return int(high_step), int(low_step)


def recruitment_to_inflation(
    eeli_high: float,
    eeli_low: float,
    tiv_low: float,
    driving_low: float,
    peep_drop: float,
    tidal_volume_low: float,
) -> tuple[float, float]:
    """R/I and recruited volume between a high and a low PEEP step (see the module).

    Args:
        eeli_high:    EELI of the high step (EIT units, global or of a region).
        eeli_low:     EELI of the low step, of the same region (EIT units).
        tiv_low:      TIV of the low step, of the same region (EIT units).
        driving_low:  Driving pressure of the low step (cmH2O).
        peep_drop:    PEEP of the high step minus PEEP of the low step (cmH2O).
        tidal_volume_low: Tidal volume of the low step, times the share of the region
                      for a region (mL); NaN if unknown.

    Returns:
        R/I, and the recruited volume in mL (NaN without a tidal volume); NaN if a value
        is missing or the low compliance or the drop of PEEP is not positive.
    """
    compliance_low = tiv_low / driving_low if driving_low > 0 else np.nan
    if not (compliance_low > 0 and peep_drop > 0):
        return np.nan, np.nan
    eeli_drop = eeli_high - eeli_low
    predicted_drop = compliance_low * peep_drop  # what the low-PEEP compliance explains
    ri = eeli_drop / predicted_drop - 1
    # In mL: each EIT unit of the low step is tidal_volume_low / tiv_low mL.
    recruited_volume = (eeli_drop - predicted_drop) * tidal_volume_low / tiv_low
    return float(ri), float(recruited_volume)
