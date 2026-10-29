"""Cardiac filtering of the pixel signals.

Policy (notebooks/05_preprocessing_choices.ipynb, section 5c):
- heart rate given by the user -> MDN filter;
- no heart rate -> only a light moving average, the cardiac component is NOT removed.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


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
