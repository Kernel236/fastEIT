"""Airway pressures of each breath: PEEP, plateau and driving pressure.

Read on the airway pressure curve, all from the same channel so that a constant offset
of the sensor cancels in the driving pressure (notebooks/06_peep_trial_segmentation.ipynb,
sections 2 and 4). The PEEP of a breath is the end-expiratory pressure at the airway
opening: the PEEP applied by the ventilator, without intrinsic PEEP (that would need an
end-expiratory occlusion), so it is a lower bound of the total PEEP. A breath whose
pressure is missing (NaN) near its start gets NaN.
"""

from __future__ import annotations

import numpy as np

from fasteit.preprocessing.cardiac_filter import moving_average_window

# Details of the method (notebook 06, sections 2 and 4), not meant to be tuned.
# Light smoothing of the pressure before taking its slope.
SMOOTH_SECONDS = 0.1
# The steepest rise is searched from 1 s before to 0.5 s after the start of the EIT breath.
ONSET_SEARCH_BEFORE_SECONDS = 1.0
ONSET_SEARCH_AFTER_SECONDS = 0.5
# Start of inspiration = last minimum in the 0.5 s before the steepest rise.
ONSET_LOOKBACK_SECONDS = 0.5
# Start of expiration = steepest fall in the 0.3 s before the pressure is back near PEEP.
EXPIRATION_SEARCH_SECONDS = 0.3
# The pressure is back near PEEP when it is within 2 (mbar or cmH2O) of it.
BACK_TO_PEEP_MARGIN = 2.0
# A plateau needs an end-inspiratory hold of at least 0.5 s (0.9-1.3 s in the trials of
# notebook 06; 5 of 606 breaths were shorter).
MIN_PAUSE_SECONDS = 0.5


def _samples(seconds: float, fs: float) -> int:
    return int(round(seconds * fs))


def _smooth(paw: np.ndarray, fs: float) -> np.ndarray:
    """Centred moving average, with the first and last value repeated at the edges."""
    window = moving_average_window(fs, SMOOTH_SECONDS)
    padded = np.pad(paw, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


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
    before = _samples(ONSET_SEARCH_BEFORE_SECONDS, fs)
    after = _samples(ONSET_SEARCH_AFTER_SECONDS, fs)
    lookback = _samples(ONSET_LOOKBACK_SECONDS, fs)

    onsets = np.full(len(eit_starts), -1, dtype=int)
    for i, start in enumerate(eit_starts):
        lo, hi = max(0, start - before), min(len(smoothed) - 1, start + after)
        if hi <= lo or np.isnan(smoothed[max(0, lo - lookback) : hi]).any():
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
    n = _samples(seconds, fs)
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
    """Plateau pressure of each breath, read at the end of the end-inspiratory hold.

    Between the start of inspiration of a breath and the next one: after the peak, the
    start of expiration is the steepest fall of the pressure in the 0.3 s before it comes
    back within 2 of the next PEEP; the plateau is the median pressure from ``window[0]``
    to ``window[1]`` seconds before it (notebook 06, section 4).

    A valid plateau needs an end-inspiratory hold of at least 0.5 s with zero flow, as
    in volume control with a pause. With a shorter hold the plateau is NaN: the reading
    would fall on the relaxation after the peak, and the peak pressure is never used
    instead because it includes the resistive pressure. In pressure control the flat
    inspiratory pressure is read, which is a plateau only if the flow is zero at the end
    of inspiration.

    Args:
        paw:    Airway pressure, one value per frame.
        fs:     Sampling frequency in Hz.
        onsets: Start of inspiration of each breath (``inspiration_onsets``).
        peeps:  PEEP of each breath (``pressure_before`` at the onsets).
        window: (start, end) of the reading, in seconds before the start of expiration.

    Returns:
        One plateau per breath; NaN for the last breath (no next breath), for unknown
        onsets, for breaths with missing pressure and where no hold is found.
    """
    paw = np.asarray(paw, dtype=float)
    smoothed = _smooth(paw, fs)
    slope = np.gradient(smoothed) * fs
    first, last = _samples(window[0], fs), _samples(window[1], fs)
    search = _samples(EXPIRATION_SEARCH_SECONDS, fs)
    min_pause = _samples(MIN_PAUSE_SECONDS, fs)

    plateaus = np.full(len(onsets), np.nan)
    for i in range(len(onsets) - 1):
        onset, next_onset = onsets[i], onsets[i + 1]
        if onset < 0 or next_onset <= onset or np.isnan(peeps[i + 1]):
            continue
        if np.isnan(paw[onset:next_onset]).any():
            continue
        peak = onset + int(np.argmax(paw[onset:next_onset]))
        back_to_peep = np.flatnonzero(
            smoothed[peak:next_onset] < peeps[i + 1] + BACK_TO_PEEP_MARGIN
        )
        if len(back_to_peep) == 0:
            continue
        reached = peak + back_to_peep[0]
        lo = max(peak, reached - search)
        expiration = lo + int(np.argmin(slope[lo : reached + 1]))
        if expiration - peak >= min_pause:
            plateaus[i] = np.median(paw[expiration - first : expiration - last])
    return plateaus
