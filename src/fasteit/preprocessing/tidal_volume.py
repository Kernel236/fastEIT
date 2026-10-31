"""Tidal volume of each breath from the ventilator curves (mL).

The inspiratory tidal volume: in volume control it is the volume delivered, the one
set on the ventilator. It is read on the volume curve, or on the flow curve by
integrating the inspiratory flow, in a window around the inspiration of each EIT
breath. With a leak (non-invasive ventilation, a leaking cuff) the inspiratory volume
includes the leak.

Tested on synthetic curves only: the recordings available have no volume or flow
curve. Inspiratory flow is taken as positive, as recorded by the ventilators.
"""

from __future__ import annotations

import numpy as np

ML_PER_L = 1000.0


def inspiration_windows(
    starts: np.ndarray, peaks: np.ndarray, ends: np.ndarray, n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Frames around the inspiration of each breath, from the EIT breath.

    From half an expiration before the start of the breath to the middle of its
    expiration: the window holds the whole inspiration even when the EIT valley falls
    a little after the start of the inspiratory flow (within about 0.1 s in
    notebooks/06_peep_trial_segmentation.ipynb, section 2), and nothing of the next
    inspiration.

    Args:
        starts, peaks, ends: Frames of start, peak and end of each breath.
        n_frames: Number of frames of the recording.

    Returns:
        First frame and frame after the last of each window.
    """
    half_expiration = (np.asarray(ends) - np.asarray(peaks)) // 2
    first = np.maximum(np.asarray(starts) - half_expiration, 0)
    after_last = np.minimum(np.asarray(peaks) + half_expiration + 1, n_frames)
    return first, after_last


def volumes_from_volume_curve(
    volume: np.ndarray, first: np.ndarray, after_last: np.ndarray
) -> np.ndarray:
    """Inspiratory tidal volume on a volume curve (mL).

    The highest volume in the window minus the lowest one before it, so that it works
    whether or not the ventilator sets the volume back to zero at each breath. If it
    does, and more volume came out than went in during the breath before, the
    difference is added (the lowest value is then the end of that expiration).

    Args:
        volume: Volume curve in mL, one value per frame.
        first, after_last: Window of each breath (``inspiration_windows``).

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

    Only the positive (inspiratory) flow of the window is integrated. A zero offset of
    the flow sensor adds offset x window length (e.g. 0.01 L/s over 4 s = 40 mL).

    Args:
        flow:  Flow curve in L/s, one value per frame, inspiration positive.
        fs:    Sampling frequency in Hz.
        first, after_last: Window of each breath (``inspiration_windows``).

    Returns:
        One volume per breath; NaN if the flow is missing in the window.
    """
    flow = np.asarray(flow, dtype=float)
    out = np.full(len(first), np.nan)
    for i, (lo, hi) in enumerate(zip(first, after_last, strict=True)):
        part = flow[lo:hi]
        if len(part) and not np.isnan(part).any():
            out[i] = np.clip(part, 0, None).sum() / fs * ML_PER_L
    return out
