"""Quality flags of the detected breaths.

Flags mark a breath, they never remove it: the clinician decides (fastEIT design
decision D5). Thresholds and the evidence for them are in
notebooks/05_preprocessing_choices.ipynb, section 8; they are fastEIT choices, not
taken from a paper.

All measures come from the global impedance signal, so Ti and Ttot are
impedance-based (valley to peak, valley to valley) and not the ventilator values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def breath_measures(signal: np.ndarray, fs: float, breaths: np.ndarray) -> pd.DataFrame:
    """Duration, Ti/Ttot and amplitude of each breath.

    The amplitude is the mean of the rise from the start valley to the peak and the
    fall from the peak to the end valley, as in eitprocessing ``BreathDetection``
    (v1.8.8; Somhorst et al., JOSS 2026;11(117):8179, DOI: 10.21105/joss.08179).

    Args:
        signal:  1-D global signal used for the breath detection.
        fs:      Sampling frequency in Hz.
        breaths: Integer array of shape (n_breaths, 3): start, peak, end frame.

    Returns:
        DataFrame with one row per breath and columns ``duration_s`` (Ttot),
        ``ti_ratio`` (Ti/Ttot) and ``amplitude`` (units of the signal).
    """
    start, peak, end = np.asarray(breaths, dtype=int).reshape(-1, 3).T
    rise = signal[peak] - signal[start]
    fall = signal[peak] - signal[end]
    return pd.DataFrame(
        {
            "duration_s": (end - start) / fs,
            "ti_ratio": (peak - start) / (end - start),
            "amplitude": (rise + fall) / 2,
        }
    )


def ratio_to_local_median(
    values: np.ndarray, window_breaths: int, min_breaths: int
) -> np.ndarray:
    """Each value divided by the median of the breaths around it.

    The window is centred on the breath and includes it. At the start and end of the
    recording the window is shorter; with fewer than ``min_breaths`` values the ratio
    is NaN, so no outlier flag can be set there.

    Args:
        values:         One value per breath, in time order (e.g. amplitudes).
        window_breaths: Number of breaths in the window (odd, e.g. 21).
        min_breaths:    Fewest breaths needed to compute the median.

    Returns:
        Array of ratios, same length as ``values``.
    """
    series = pd.Series(values, dtype=float)
    local_median = series.rolling(
        window_breaths, center=True, min_periods=min_breaths
    ).median()
    return (series / local_median).to_numpy()
