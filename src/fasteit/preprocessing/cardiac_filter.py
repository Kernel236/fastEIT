"""Cardiac filtering of the pixel signals.

Policy (notebooks/05_preprocessing_choices.ipynb, section 5c):
- heart rate given by the user -> MDN filter;
- no heart rate -> only a light moving average, the cardiac component is NOT removed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

SECONDS_PER_MINUTE = 60.0


def moving_average_window(fs: float, seconds: float) -> int:
    """Number of samples of a centred moving average lasting ``seconds``.

    The number is odd so that the window is centred on each sample.

    Args:
        fs:      Sampling frequency in Hz.
        seconds: Window duration in seconds.

    Returns:
        Odd number of samples, at least 1 (5 for 0.1 s at 50 Hz, 3 at 20 Hz).
    """
    samples = int(round(seconds * fs))
    odd_samples = samples // 2 * 2 + 1
    return max(1, odd_samples)


def moving_average(signals: np.ndarray, fs: float, seconds: float = 0.1) -> np.ndarray:
    """Centred moving average along the time axis (axis 0).

    Fallback used when no heart rate is available (section 5c of the evidence
    notebook). Same idea as the MATLAB ``smooth()`` of the original analysis
    scripts, but with the width in seconds. At 50 Hz a 0.1 s window keeps about
    96 % of a 100/min cardiac oscillation, so it is a smoothing, not a cardiac filter.

    Args:
        signals: Array with time on axis 0, e.g. pixels of shape (N, 32, 32) or a
                 1-D global signal of shape (N,).
        fs:      Sampling frequency in Hz.
        seconds: Window duration in seconds.

    Returns:
        Array of the same shape as ``signals``. At the edges the first and last
        sample are repeated, so the length does not change.
    """
    window = moving_average_window(fs, seconds)
    return ndimage.uniform_filter1d(
        np.asarray(signals, dtype=float), size=window, axis=0, mode="nearest"
    )


def respiratory_rate(
    global_signal: np.ndarray,
    fs: float,
    min_bpm: float = 6.0,
    max_bpm: float = 60.0,
    window_seconds: float = 30.0,
) -> float:
    """Respiratory rate (Hz) from the power spectrum of the global signal.

    Same method as the respiratory part of eitprocessing ``RateDetection``
    (v1.8.8; Somhorst et al., JOSS 2026, DOI: 10.21105/joss.08179): Welch
    spectrum (Hann window, 50 % overlap), highest peak between ``min_bpm`` and
    ``max_bpm``, refined with a parabola through the peak and its two neighbours.
    In the evidence notebook (section 4) it agrees with the device breath markers
    within 1.7 breaths/min.

    Args:
        global_signal:  1-D signal, e.g. the sum of all pixels per frame.
        fs:             Sampling frequency in Hz.
        min_bpm:        Lowest respiratory rate searched, breaths/min.
        max_bpm:        Highest respiratory rate searched, breaths/min.
        window_seconds: Welch window; shortened to the signal length if needed.

    Returns:
        Respiratory rate in Hz (multiply by 60 for breaths/min).
    """
    x = np.asarray(global_signal, dtype=float)
    segment = min(int(window_seconds * fs), len(x))
    frequencies, power = signal.welch(
        x,
        fs,
        window=signal.windows.hann(segment, sym=False),
        nperseg=segment,
        noverlap=int(segment * 0.5),
        detrend="constant",
    )

    in_range = (frequencies >= min_bpm / SECONDS_PER_MINUTE) & (
        frequencies <= max_bpm / SECONDS_PER_MINUTE
    )
    range_frequencies, range_power = frequencies[in_range], power[in_range]
    peak = int(np.argmax(range_power))

    # A peak on the edge of the range cannot be refined: return the grid value.
    if peak == 0 or peak == len(range_power) - 1:
        return float(range_frequencies[peak])

    # Parabolic interpolation through the peak and its two neighbours.
    left, centre, right = range_power[peak - 1 : peak + 2]
    step = range_frequencies[peak + 1] - range_frequencies[peak]
    curvature = left - 2 * centre + right
    shift = 0.5 * (left - right) / curvature if curvature != 0 else 0.0
    return float(range_frequencies[peak] + shift * step)
