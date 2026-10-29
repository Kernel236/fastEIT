"""Quality flags of the detected breaths.

Flags mark a breath, they never remove it: the clinician decides (fastEIT design
decision D5). Thresholds and the evidence for them are in
notebooks/05_preprocessing_choices.ipynb, section 8; they are fastEIT choices, not
taken from a paper.

All measures come from the global impedance signal, so Ti and Ttot are
impedance-based (valley to peak, valley to valley) and not the ventilator values;
the amplitude is in arbitrary impedance units, not a volume.
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
        ``ti_ratio`` (Ti/Ttot) and ``amplitude`` (units of the signal). A breath of
        zero duration has ``ti_ratio`` NaN.

    Raises:
        ValueError: if ``breaths`` is not an array of shape (n_breaths, 3).
    """
    signal = np.asarray(signal, dtype=float)
    breaths = np.asarray(breaths, dtype=int)
    if breaths.ndim != 2 or breaths.shape[1] != 3:
        raise ValueError(
            f"breaths must have shape (n_breaths, 3), got {breaths.shape}."
        )
    start, peak, end = breaths.T
    rise = signal[peak] - signal[start]
    fall = signal[peak] - signal[end]
    with np.errstate(divide="ignore", invalid="ignore"):  # zero duration -> NaN
        ti_ratio = (peak - start) / (end - start)
    return pd.DataFrame(
        {
            "duration_s": (end - start) / fs,
            "ti_ratio": ti_ratio,
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


def quality_flags(
    signal: np.ndarray,
    fs: float,
    breaths: np.ndarray,
    near_invalid: np.ndarray,
    short_seconds: float,
    long_seconds: float,
    ti_ratio_low: float,
    ti_ratio_high: float,
    window_breaths: int,
    min_breaths: int,
    amplitude_ratio_low: float,
    amplitude_ratio_high: float,
    duration_ratio_high: float,
) -> pd.DataFrame:
    """One true/false column per quality check, one row per breath.

    - ``flag_short`` / ``flag_long``: duration below ``short_seconds`` or above
      ``long_seconds``;
    - ``flag_suspect_ti_ratio``: Ti/Ttot outside ``ti_ratio_low``-``ti_ratio_high``;
    - ``flag_amplitude_outlier``: amplitude below ``amplitude_ratio_low`` or above
      ``amplitude_ratio_high`` times the median of the surrounding breaths;
    - ``flag_duration_outlier``: duration above ``duration_ratio_high`` times the
      median of the surrounding breaths;
    - ``flag_invalid_data``: close to invalid data (``near_invalid`` of
      ``detect_breaths``).

    On 946 breaths of 11 regular recordings these rules flagged 9 Ti/Ttot, 1 amplitude
    and 3 duration outliers, all visible events (section 8 of the evidence notebook).

    Args:
        signal:       1-D global signal used for the breath detection.
        fs:           Sampling frequency in Hz.
        breaths:      Integer array (n_breaths, 3) from ``detect_breaths``.
        near_invalid: Boolean array (n_breaths,) from ``detect_breaths``.
        short_seconds, long_seconds: Duration limits in seconds.
        ti_ratio_low, ti_ratio_high: Ti/Ttot limits.
        window_breaths, min_breaths: Window of ``ratio_to_local_median``.
        amplitude_ratio_low, amplitude_ratio_high: Amplitude limits as ratios to the
            local median.
        duration_ratio_high: Duration limit as a ratio to the local median.

    Returns:
        DataFrame of booleans, one column per flag. The pipeline adds whether the
        breath is excluded and ``is_valid`` (no flag and not excluded).

    Raises:
        ValueError: if ``breaths`` and ``near_invalid`` have different lengths.
    """
    measures = breath_measures(signal, fs, breaths)  # also checks the shape
    if len(near_invalid) != len(measures):
        raise ValueError(
            f"near_invalid has {len(near_invalid)} values for {len(measures)} breaths."
        )
    amplitude_ratio = ratio_to_local_median(
        measures.amplitude, window_breaths, min_breaths
    )
    duration_ratio = ratio_to_local_median(
        measures.duration_s, window_breaths, min_breaths
    )
    # A NaN ratio (too few breaths around) compares as False: no flag.
    return pd.DataFrame(
        {
            "flag_short": measures.duration_s < short_seconds,
            "flag_long": measures.duration_s > long_seconds,
            "flag_suspect_ti_ratio": (measures.ti_ratio < ti_ratio_low)
            | (measures.ti_ratio > ti_ratio_high),
            "flag_amplitude_outlier": (amplitude_ratio < amplitude_ratio_low)
            | (amplitude_ratio > amplitude_ratio_high),
            "flag_duration_outlier": duration_ratio > duration_ratio_high,
            "flag_invalid_data": np.asarray(near_invalid, dtype=bool),
        }
    )
