"""Breath detection on the global impedance signal.

Reimplementation of eitprocessing ``BreathDetection`` (v1.8.8; Somhorst et al.,
JOSS 2026;11(117):8179, DOI: 10.21105/joss.08179), checked to give the same breaths
on 11 real recordings (notebooks/05_preprocessing_choices.ipynb, section 8).

One change: breaths close to invalid data are flagged, not removed.

A breath is a triplet of frame indices: start (valley), peak, end (next valley).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def invalid_samples(
    signal: np.ndarray, percentile: float, multiplier: float
) -> np.ndarray:
    """Indices of the samples too far from the mean to be breathing (step 1).

    A sample is invalid when it is below ``mean - multiplier * (mean - low)`` or above
    ``mean + multiplier * (high - mean)``, where ``low`` and ``high`` are the
    ``percentile`` and ``100 - percentile`` percentiles of the signal. With the
    defaults (5, 4) only gross outliers are caught, e.g. a disconnection spike.

    Args:
        signal:     1-D global signal.
        percentile: Lower percentile; the upper one is ``100 - percentile``.
        multiplier: How many times the mean-to-percentile distance is allowed.

    Returns:
        Sorted indices of the invalid samples (empty if none).
    """
    mean = np.mean(signal)
    low = np.percentile(signal, percentile)
    high = np.percentile(signal, 100 - percentile)
    lower_cutoff = mean - (mean - low) * multiplier
    upper_cutoff = mean + (high - mean) * multiplier
    return np.flatnonzero((signal < lower_cutoff) | (signal > upper_cutoff))


def fill_with_nearest(signal: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Replace the samples at ``missing`` with the nearest valid sample (step 1).

    In a run of missing samples the first half takes the value just before the run
    and the second half the value just after it. A run at the start or at the end of
    the signal takes the only valid neighbour.

    Args:
        signal:  1-D signal.
        missing: Indices of the samples to replace.

    Returns:
        A copy of ``signal`` without the missing values.

    Raises:
        ValueError: if every sample is missing.
    """
    filled = np.array(signal, dtype=float)
    missing = np.unique(missing)
    if len(missing) == 0:
        return filled
    if len(missing) == len(filled):
        raise ValueError("Every sample is missing: nothing to fill with.")

    # Split the missing indices into runs of consecutive samples.
    runs = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
    for run in runs:
        first, last = run[0], run[-1]
        if first == 0:
            filled[run] = filled[last + 1]
        elif last == len(filled) - 1:
            filled[run] = filled[first - 1]
        else:
            half = len(run) // 2
            filled[run[:half]] = filled[first - 1]
            filled[run[half:]] = filled[last + 1]
    return filled


def local_baseline(signal: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    """Slow moving average of the signal, used as a local threshold (step 2).

    Weighted with a Blackman window, centred, with the first and last value repeated at
    the edges. The window has an odd number of samples and is shortened to fit a signal
    shorter than ``seconds``.

    Args:
        signal:  1-D signal.
        fs:      Sampling frequency in Hz.
        seconds: Window width in seconds.

    Returns:
        Array of the same length as ``signal``.
    """
    window = int(fs * seconds)
    if window % 2 == 0:
        window += 1
    if window > len(signal):
        window = (len(signal) - 1) // 2 * 2 + 1
    weights = np.blackman(window)
    weights = weights / weights.sum()
    padded = np.pad(signal, (window - 1) // 2, mode="edge")
    return np.convolve(padded, weights, mode="valid")


def peaks_and_valleys(
    signal: np.ndarray, baseline: np.ndarray, fs: float, min_duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate peaks above the baseline and valleys below it (step 3).

    Args:
        signal:       1-D signal.
        baseline:     Local threshold of the same length (``local_baseline``).
        fs:           Sampling frequency in Hz.
        min_duration: Minimum time in seconds between two peaks and between two valleys.

    Returns:
        Indices of the peaks and indices of the valleys.
    """
    distance = max(min_duration * fs, 1)
    peaks, _ = find_peaks(signal, distance=distance, height=baseline)
    valleys, _ = find_peaks(-signal, distance=distance, height=-baseline)
    return peaks, valleys
