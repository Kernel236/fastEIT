"""Breath detection on the global impedance signal.

Reimplementation of eitprocessing ``BreathDetection`` (v1.8.8; Somhorst et al.,
JOSS 2026;11(117):8179, DOI: 10.21105/joss.08179). It gives the same breaths as
eitprocessing on 11 regular real recordings (notebooks/05_preprocessing_choices.ipynb,
section 8) and on synthetic signals with a spike and with small breaths. This shows
equivalence with eitprocessing, not accuracy against a gold standard.

One change: breaths close to invalid data are flagged, not removed.

A breath is a triplet of frame indices: start (valley), peak, end (next valley).
Impedance is assumed to rise with inspiration (as in the Draeger .bin pixels).

Limitations (to keep in mind when reading Ti, Te and the respiratory rate):
- the valley is the minimum of impedance, i.e. of lung volume, not the start of
  the inspiratory effort or the ventilator trigger; the peak includes any
  inspiratory pause. Timings are impedance-based and not interchangeable with the
  ventilator values;
- patient-ventilator asynchrony is not detected: an ineffective effort is usually
  not a peak, and two inspirations without a valley between them become one breath;
- the 15 s baseline follows slow EELI drift, but after a sudden EELI step (PEEP
  change, recruitment) the breaths within about 15 s of the step are unreliable;
- without a heart rate the cardiac component is not removed, so small breaths and
  the exact position of valleys and peaks are less reliable.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks


def invalid_samples(
    signal: np.ndarray, percentile: float, multiplier: float
) -> np.ndarray:
    """Indices of the gross outliers of the signal (step 1).

    A sample is invalid when it is below ``mean - multiplier * (mean - low)`` or above
    ``mean + multiplier * (high - mean)``, where ``low`` and ``high`` are the
    ``percentile`` and ``100 - percentile`` percentiles of the signal. With the
    defaults (5, 4) only gross outliers are caught: artefacts such as a disconnection
    spike, but also extreme manoeuvres such as a sustained inflation or a large
    recruitment step, which are physiological.

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

    In a run of missing samples the first ``len(run) // 2`` samples take the value
    just before the run and the others the value just after it (so the middle sample
    of an odd run takes the value after). A run at the start or at the end of the
    signal takes the only valid neighbour.

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


def trim_edges(
    signal: np.ndarray, peaks: np.ndarray, valleys: np.ndarray, baseline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the incomplete breaths at the start and end of the signal (step 4).

    The first valley is dropped if the signal before it never rises above the baseline
    (the recording started in the middle of an expiration), and the same for the last
    valley. Then only the peaks between the first and the last valley are kept.

    Args:
        signal:   1-D signal.
        peaks:    Candidate peak indices.
        valleys:  Candidate valley indices, never the first or last sample (true for
                  the output of ``peaks_and_valleys``).
        baseline: Local threshold of the same length as ``signal``.

    Returns:
        Peaks and valleys without the edge cases (both empty if no breath is left).
    """
    if len(valleys) == 0:
        return peaks, valleys
    if signal[: valleys[0]].max() < baseline[valleys[0]]:
        valleys = valleys[1:]
    if len(valleys) and signal[valleys[-1] :].max() < baseline[valleys[-1]]:
        valleys = valleys[:-1]
    if len(valleys) == 0:
        return peaks[:0], valleys
    peaks = peaks[(peaks > valleys[0]) & (peaks < valleys[-1])]
    return peaks, valleys


def drop_doubles(
    signal: np.ndarray, peaks: np.ndarray, valleys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one peak between two valleys and one valley between two peaks (step 5).

    Walking from the first valley: if two valleys have no peak between them, the
    higher valley is dropped; if two valleys have more than one peak between them, the
    lower of the first two peaks is dropped. Repeat until every pair of consecutive
    valleys holds exactly one peak.

    Args:
        signal:  1-D signal.
        peaks:   Peak indices.
        valleys: Valley indices.

    Returns:
        Peaks and valleys, alternating.
    """
    peaks, valleys = list(peaks), list(valleys)
    i = 0
    while i < len(valleys) - 1:
        between = [k for k, p in enumerate(peaks) if valleys[i] < p < valleys[i + 1]]
        if not between:
            higher = i if signal[valleys[i]] > signal[valleys[i + 1]] else i + 1
            del valleys[higher]
            continue
        if len(between) > 1:
            first, second = between[0], between[1]
            lower = first if signal[peaks[first]] < signal[peaks[second]] else second
            del peaks[lower]
            continue
        i += 1
    return np.array(peaks, dtype=int), np.array(valleys, dtype=int)


def drop_low_amplitude(
    signal: np.ndarray, peaks: np.ndarray, valleys: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge peaks that are too small to be a breath into the breath around them (step 6).

    The amplitude of a peak is the mean of its rise from the valley before and its fall
    to the valley after. Peaks below ``fraction`` times the median amplitude are
    removed, then ``drop_doubles`` keeps the lower of the two valleys left next to each
    other, so the small peak becomes part of a longer breath.

    Args:
        signal:   1-D signal.
        peaks:    Peak indices, alternating with ``valleys`` (one more valley than peaks).
        valleys:  Valley indices.
        fraction: Fraction of the median amplitude below which a peak is removed.

    Returns:
        Peaks and valleys, alternating.

    Raises:
        ValueError: if there is not exactly one more valley than peaks.
    """
    if len(peaks) == 0 or len(valleys) == 0:
        return peaks, valleys
    if len(valleys) != len(peaks) + 1:
        raise ValueError(
            "Peaks and valleys must alternate, starting and ending with a valley "
            "(run drop_doubles first)."
        )
    rise = signal[peaks] - signal[valleys[:-1]]
    fall = signal[peaks] - signal[valleys[1:]]
    amplitude = (rise + fall) / 2
    peaks = peaks[amplitude >= fraction * np.median(amplitude)]
    return drop_doubles(signal, peaks, valleys)


def flag_near_invalid(
    breaths: np.ndarray, invalid: np.ndarray, margin: int
) -> np.ndarray:
    """Which breaths are close to an invalid sample (step 8).

    Same window as eitprocessing: from ``start - margin`` (included) to
    ``end + margin`` (excluded), like a Python slice. eitprocessing removes these
    breaths; fastEIT keeps them with a flag.

    Args:
        breaths: Integer array of shape (n_breaths, 3): start, peak, end frame.
        invalid: Indices of the invalid samples (``invalid_samples``).
        margin:  Margin in samples.

    Returns:
        Boolean array of length n_breaths.
    """
    return np.array(
        [
            np.any((invalid >= start - margin) & (invalid < end + margin))
            for start, _, end in breaths
        ],
        dtype=bool,
    )


def detect_breaths(
    signal: np.ndarray,
    fs: float,
    invalid_percentile: float,
    invalid_multiplier: float,
    baseline_seconds: float,
    min_duration_seconds: float,
    low_amplitude_fraction: float,
    invalid_margin_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the breaths in the global signal (steps 1-8 of eitprocessing BreathDetection).

    1. invalid samples are replaced with the nearest valid value;
    2. a slow Blackman moving average is the local baseline;
    3. peaks above it and valleys below it, at least ``min_duration_seconds`` apart;
    4. incomplete breaths at the edges are dropped;
    5. one peak between two valleys, one valley between two peaks;
    6. too small peaks are merged into the breath around them;
    7. breaths = valley, peak, next valley;
    8. breaths closer than ``invalid_margin_seconds`` to invalid data are flagged
       (eitprocessing removes them).

    Args:
        signal:                 1-D global signal without NaN (after the cardiac filter).
        fs:                     Sampling frequency in Hz.
        invalid_percentile:     See ``invalid_samples``.
        invalid_multiplier:     See ``invalid_samples``.
        baseline_seconds:       Width of the local baseline.
        min_duration_seconds:   Minimum time between two peaks and between two valleys.
        low_amplitude_fraction: See ``drop_low_amplitude``.
        invalid_margin_seconds: Margin around invalid samples for the flag.

    Returns:
        ``breaths``: integer array of shape (n_breaths, 3) with the start, peak and end
        frame of each breath; ``near_invalid``: boolean array of length n_breaths.

    Raises:
        ValueError: if ``fs`` is not positive, the signal is not a non-empty 1-D array
            or it contains NaN or inf.
    """
    if not fs > 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError("Global signal must be a non-empty 1-D array.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Global signal contains NaN or inf.")
    no_breaths = (np.empty((0, 3), dtype=int), np.empty(0, dtype=bool))

    # A constant signal has no breaths. Checked here because rounding in the mean can
    # make every sample of a constant signal "invalid" (eitprocessing then fails).
    if np.ptp(signal) == 0:
        return no_breaths

    invalid = invalid_samples(signal, invalid_percentile, invalid_multiplier)
    data = fill_with_nearest(signal, invalid)
    baseline = local_baseline(data, fs, baseline_seconds)
    peaks, valleys = peaks_and_valleys(data, baseline, fs, min_duration_seconds)
    if len(valleys) < 2 or len(peaks) < 1:
        return no_breaths

    peaks, valleys = trim_edges(data, peaks, valleys, baseline)
    peaks, valleys = drop_doubles(data, peaks, valleys)
    peaks, valleys = drop_low_amplitude(data, peaks, valleys, low_amplitude_fraction)
    if len(peaks) == 0:
        return no_breaths

    breaths = np.column_stack([valleys[:-1], peaks, valleys[1:]])
    margin = math.ceil(invalid_margin_seconds * fs)
    return breaths, flag_near_invalid(breaths, invalid, margin)
