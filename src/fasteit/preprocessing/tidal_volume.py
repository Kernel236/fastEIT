"""Tidal volume of each breath from the ventilator curves (mL).

The inspiratory tidal volume: in volume control it is the volume delivered, the one
set on the ventilator. It is read on the volume curve, or on the flow curve by
integrating the inspiratory flow, in a window around the inspiration of each EIT
breath. It is the volume measured by the ventilator: with a leak (non-invasive
ventilation, a leaking cuff) it includes the leak, and without compensation of the
compressible volume of the circuit it includes that too.

Tested on synthetic curves only: the recordings available have no volume or flow
curve. Inspiratory flow is taken as positive, as recorded by the ventilators.
"""

from __future__ import annotations

import numpy as np

from fasteit.preprocessing.auxiliary import ML_PER_L

# A tidal volume typed by the user outside this range gives a warning: a plausibility
# check for adults (e.g. litres typed instead of mL), not a clinical limit.
TYPED_VOLUME_RANGE_ML = (100.0, 1500.0)


def inspiration_windows(
    starts: np.ndarray, peaks: np.ndarray, ends: np.ndarray, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Frames around the inspiration of each breath, from the EIT breath.

    From half an expiration before the start of the breath (but not before the middle
    of the expiration of the breath before) to the middle of its own expiration. The
    window holds the whole inspiration even when the EIT valley falls a little after
    the start of the inspiratory flow (within about 0.1 s in
    notebooks/06_peep_trial_segmentation.ipynb, section 2), and nothing of the
    inspirations before and after it, also after a long inspiratory hold.

    Args:
        starts:   Frame of the start of each breath.
        peaks:    Frame of the peak of each breath.
        ends:     Frame of the end of each breath.
        n_frames: Number of frames of the recording.

    Returns:
        First frame and frame after the last of each window.
    """
    starts, peaks, ends = np.asarray(starts), np.asarray(peaks), np.asarray(ends)
    half_expiration = (ends - peaks) // 2

    # Middle of the expiration of the breath before (frame 0 for the first breath).
    previous_middle = np.zeros(len(starts), dtype=int)
    previous_middle[1:] = (peaks[:-1] + starts[1:]) // 2

    # From half an expiration before the start, but not before that middle...
    first = np.maximum(starts - half_expiration, previous_middle)
    # ...to the middle of its own expiration, inside the recording.
    after_last = np.minimum(peaks + half_expiration + 1, n_frames)
    return first, after_last


def volumes_from_volume_curve(
    volume: np.ndarray, first: np.ndarray, after_last: np.ndarray
) -> np.ndarray:
    """Inspiratory tidal volume on a volume curve (mL).

    The highest volume in the window minus the lowest one before it, so that it works
    whether or not the ventilator sets the volume back to zero at each breath. If it
    does, and more volume came out than went in during the breath before (e.g. the
    first breaths after a decrease of PEEP, as the lung empties), the difference is
    added: the lowest value is then the end of that expiration.

    Args:
        volume:     Volume curve in mL, one value per frame.
        first:      First frame of the window of each breath (``inspiration_windows``).
        after_last: Frame after the last of each window.

    Returns:
        One volume per breath; NaN if the volume is missing in the window.
    """
    volume = np.asarray(volume, dtype=float)
    out = np.full(len(first), np.nan)
    for i, (lo, hi) in enumerate(zip(first, after_last, strict=True)):
        part = volume[lo:hi]
        if len(part) and not np.isnan(part).any():
            top = int(np.argmax(part))
            out[i] = part[top] - part[: top + 1].min()
    return out


def volumes_from_flow_curve(
    flow: np.ndarray, fs: float, first: np.ndarray, after_last: np.ndarray
) -> np.ndarray:
    """Inspiratory tidal volume on a flow curve (mL): integral of the inspiratory flow.

    The flow is integrated over the inspiration: the run of positive flow around its
    highest value in the window. Noise and cardiac oscillations around zero flow
    (end-inspiratory pause, end of expiration) are left out, unlike a sum of all the
    positive flow of the window. A zero offset of the flow sensor adds offset x
    duration of the inspiration (e.g. 0.01 L/s over 1 s = 10 mL).

    Args:
        flow:       Flow curve in L/s, one value per frame, inspiration positive.
        fs:         Sampling frequency in Hz.
        first:      First frame of the window of each breath (``inspiration_windows``).
        after_last: Frame after the last of each window.

    Returns:
        One volume per breath; NaN if the flow is missing in the window, 0 if it is
        never positive.
    """
    flow = np.asarray(flow, dtype=float)
    out = np.full(len(first), np.nan)
    for i, (lo, hi) in enumerate(zip(first, after_last, strict=True)):
        part = flow[lo:hi]
        if len(part) == 0 or np.isnan(part).any():
            continue
        top = int(np.argmax(part))
        if part[top] <= 0:
            out[i] = 0.0
            continue
        not_inspiring = np.flatnonzero(part <= 0)
        before = not_inspiring[not_inspiring < top]
        after = not_inspiring[not_inspiring > top]
        start = before[-1] + 1 if len(before) else 0
        end = after[0] if len(after) else len(part)
        out[i] = part[start:end].sum() / fs * ML_PER_L
    return out
