"""Airway pressures of each breath: PEEP, plateau and driving pressure.

Read on the airway pressure curve, all from the same channel so that a constant offset
of the sensor cancels in the driving pressure (notebooks/06_peep_trial_segmentation.ipynb,
sections 2 and 4). The PEEP read at the airway opening is the applied PEEP: intrinsic
PEEP would need an end-expiratory occlusion. A breath whose pressure is missing (NaN)
near its start gets NaN.
"""

from __future__ import annotations

import numpy as np

from fasteit.preprocessing.cardiac_filter import moving_average_window

# Details of the method (notebook 06, sections 2 and 4), not meant to be tuned.
SMOOTH_SECONDS = 0.1  # light smoothing of the pressure before taking its slope
ONSET_SEARCH_SECONDS = (
    1.0,
    0.5,
)  # steepest rise searched from 1 s before to 0.5 s after the EIT start
ONSET_LOOKBACK_SECONDS = (
    0.5  # start of inspiration = last minimum in the 0.5 s before the steepest rise
)
EXPIRATION_SEARCH_SECONDS = (
    0.3  # start of expiration = steepest fall in the 0.3 s before the PEEP is reached
)
BACK_TO_PEEP = (
    2.0  # the PEEP is reached when the pressure is within 2 (mbar or cmH2O) of it
)


def _smooth(paw: np.ndarray, fs: float) -> np.ndarray:
    window = moving_average_window(fs, SMOOTH_SECONDS)
    return np.convolve(paw, np.ones(window) / window, mode="same")


def inspiration_onsets(
    paw: np.ndarray, fs: float, eit_starts: np.ndarray
) -> np.ndarray:
    """Start of inspiration on the pressure curve, near the start of each EIT breath.

    The steepest rise of the pressure is searched from 1 s before to 0.5 s after the
    start of the EIT breath; the start of inspiration is the last minimum of the pressure
    in the 0.5 s before it. The EIT valley falls within about 0.1 s of it, but there the
    pressure already rises quickly (notebook 06, section 2).

    Args:
        paw:        Airway pressure, one value per frame.
        fs:         Sampling frequency in Hz.
        eit_starts: Frame of the start of each EIT breath (``breaths.start_idx``).

    Returns:
        Frame of the start of inspiration of each breath; -1 where the pressure is
        missing in the search window.
    """
    smoothed = _smooth(np.asarray(paw, dtype=float), fs)
    slope = np.gradient(smoothed) * fs
    before = int(ONSET_SEARCH_SECONDS[0] * fs)
    after = int(ONSET_SEARCH_SECONDS[1] * fs)
    lookback = int(ONSET_LOOKBACK_SECONDS * fs)

    onsets = np.full(len(eit_starts), -1, dtype=int)
    for i, start in enumerate(eit_starts):
        lo, hi = max(0, start - before), min(len(smoothed) - 1, start + after)
        if hi <= lo or np.isnan(slope[lo:hi]).any():
            continue
        steepest = lo + int(np.argmax(slope[lo:hi]))
        back = max(0, steepest - lookback)
        before_rise = smoothed[back : steepest + 1]
        # Last occurrence of the minimum (on a flat PEEP the first one would be too early).
        onsets[i] = back + len(before_rise) - 1 - int(np.argmin(before_rise[::-1]))
    return onsets


def pressure_before(
    paw: np.ndarray, fs: float, frames: np.ndarray, seconds: float
) -> np.ndarray:
    """Median pressure in the ``seconds`` before each frame (the PEEP of a breath).

    Args:
        paw:     Airway pressure, one value per frame.
        fs:      Sampling frequency in Hz.
        frames:  Frames, e.g. the starts of inspiration (-1 = unknown).
        seconds: Length of the reading.

    Returns:
        One median per frame; NaN if the frame is unknown, too close to the start of the
        recording, or the pressure is missing in the reading.
    """
    paw = np.asarray(paw, dtype=float)
    n = int(round(seconds * fs))
    values = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        if frame - n >= 0:
            values[i] = np.median(paw[frame - n : frame])
    return values


def plateau_pressures(
    paw: np.ndarray,
    fs: float,
    onsets: np.ndarray,
    peeps: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """Plateau pressure of each breath, read at the end of the end-inspiratory pause.

    Between the start of inspiration of a breath and the next one: after the peak, the
    start of expiration is the steepest fall of the pressure in the 0.3 s before it comes
    back within 2 of the next PEEP; the plateau is the median pressure from ``window[0]``
    to ``window[1]`` seconds before it (notebook 06, section 4). Without a pause the
    reading would start before the peak, and the plateau is NaN: the peak pressure must
    not be used instead, because it includes the resistive pressure.

    Args:
        paw:    Airway pressure, one value per frame.
        fs:     Sampling frequency in Hz.
        onsets: Start of inspiration of each breath (``inspiration_onsets``).
        peeps:  PEEP of each breath (``pressure_before`` at the onsets).
        window: (start, end) of the reading, in seconds before the start of expiration.

    Returns:
        One plateau per breath; NaN for the last breath (no next breath), for unknown
        onsets and where no pause is found.
    """
    paw = np.asarray(paw, dtype=float)
    smoothed = _smooth(paw, fs)
    slope = np.gradient(smoothed) * fs
    first, last = int(window[0] * fs), int(window[1] * fs)
    search = int(EXPIRATION_SEARCH_SECONDS * fs)

    plateaus = np.full(len(onsets), np.nan)
    for i in range(len(onsets) - 1):
        onset, next_onset = onsets[i], onsets[i + 1]
        if onset < 0 or next_onset <= onset or np.isnan(peeps[i + 1]):
            continue
        peak = onset + int(np.nanargmax(paw[onset:next_onset]))
        back_to_peep = np.flatnonzero(
            smoothed[peak:next_onset] < peeps[i + 1] + BACK_TO_PEEP
        )
        if len(back_to_peep) == 0:
            continue
        reached = peak + back_to_peep[0]
        lo = max(peak, reached - search)
        expiration = lo + int(np.argmin(slope[lo : reached + 1]))
        if expiration - first > peak:
            plateaus[i] = np.median(paw[expiration - first : expiration - last])
    return plateaus
