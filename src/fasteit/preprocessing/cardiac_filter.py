"""Cardiac filtering of the pixel signals.

Policy (notebooks/05_preprocessing_choices.ipynb, section 5c):
- heart rate given by the user -> MDN filter;
- no heart rate -> only a light moving average, the cardiac component is NOT removed.

All functions work along axis 0 (time). Rates are in Hz here; the configuration uses
breaths or beats per minute, and the caller converts with ``SECONDS_PER_MINUTE``.
Parameters have no defaults here on purpose: they live only in
``fasteit.config.PreprocessingConfig``.

Valid for adult recordings only (respiratory rate 6-60/min); not for neonates,
paediatric patients or high-frequency oscillation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

SECONDS_PER_MINUTE = 60.0


def _check_sampling_frequency(fs: float) -> None:
    if not fs > 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")


def _check_no_gaps_in_time(signals: np.ndarray) -> None:
    """Reject signals that are NaN or infinite at only some frames.

    One NaN frame would spread along the whole series in the filters below. A pixel
    that is NaN for the whole recording is allowed: the filters work pixel by pixel,
    so it stays NaN and does not affect the other pixels.
    """
    not_finite = ~np.isfinite(signals)
    partly_missing = not_finite.any(axis=0) & ~not_finite.all(axis=0)
    if np.any(partly_missing):
        raise ValueError(
            "Signal has NaN or inf at some frames only: fill or exclude those frames "
            "before filtering."
        )


def _sosfiltfilt_padding(sos: np.ndarray) -> int:
    """Default padding length of ``scipy.signal.sosfiltfilt`` (same formula as scipy)."""
    zeros = min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return int(3 * (2 * len(sos) + 1 - zeros))


def moving_average_window(fs: float, seconds: float) -> int:
    """Number of samples of a centred moving average lasting ``seconds``.

    The number is odd so that the window is centred on each sample, so the actual
    duration can differ a little from ``seconds`` (0.1 s gives 5 samples at 50 Hz,
    i.e. 0.1 s, but 3 samples at 20 Hz, i.e. 0.15 s).

    Args:
        fs:      Sampling frequency in Hz.
        seconds: Window duration in seconds.

    Returns:
        Odd number of samples, at least 1.
    """
    _check_sampling_frequency(fs)
    samples = int(round(seconds * fs))
    odd_samples = samples // 2 * 2 + 1
    return max(1, odd_samples)


def moving_average(signals: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    """Centred moving average along the time axis (axis 0).

    Fallback used when no heart rate is available (section 5c of the evidence
    notebook). Same idea as the MATLAB ``smooth()`` of the original analysis
    scripts, but with the width in seconds. It is a smoothing, not a cardiac filter:
    a 100/min cardiac oscillation keeps about 96 % of its amplitude at 50 Hz and
    about 91 % at 20 Hz with a 0.1 s window.

    Args:
        signals: Array with time on axis 0, e.g. pixels of shape (N, 32, 32) or a
                 1-D global signal of shape (N,).
        fs:      Sampling frequency in Hz.
        seconds: Window duration in seconds.

    Returns:
        Array of the same shape as ``signals``. At the edges the first and last
        sample are repeated, so the length does not change.

    Raises:
        ValueError: if ``fs`` is not positive or a signal has NaN at some frames only.
    """
    signals = np.asarray(signals, dtype=float)
    _check_no_gaps_in_time(signals)
    window = moving_average_window(fs, seconds)
    return ndimage.uniform_filter1d(signals, size=window, axis=0, mode="nearest")


def respiratory_rate(
    global_signal: np.ndarray,
    fs: float,
    min_bpm: float,
    max_bpm: float,
    window_seconds: float,
) -> float:
    """Respiratory rate (Hz) from the power spectrum of the global signal.

    Same method as the respiratory part of eitprocessing ``RateDetection``
    (v1.8.8; Somhorst et al., JOSS 2026;11(117):8179, DOI: 10.21105/joss.08179):
    Welch spectrum (Hann window, 50 % overlap, mean removed), highest peak between
    ``min_bpm`` and ``max_bpm``, refined with a parabola through the peak and its two
    neighbours.

    It is the mean rate of the whole signal, not of single breaths. In the evidence
    notebook (section 4) the largest difference from the device breath markers was
    1.7 breaths/min over 11 recordings (rates 10-42/min). With a 30 s window the
    spectral bins are 2/min apart before the refinement.

    Args:
        global_signal:  1-D signal without NaN, e.g. the sum of all pixels per frame.
        fs:             Sampling frequency in Hz.
        min_bpm:        Lowest respiratory rate searched, breaths/min.
        max_bpm:        Highest respiratory rate searched, breaths/min.
        window_seconds: Welch window; shortened to the signal length if needed.

    Returns:
        Respiratory rate in Hz (multiply by 60 for breaths/min). A peak on the edge of
        the range is returned unrefined; it usually means a slow drift of the signal,
        not a real respiratory rate.

    Raises:
        ValueError: if the input is not usable (non-positive ``fs``, ``min_bpm`` not
            below ``max_bpm``, NaN in the signal, flat signal, or signal shorter than
            two cycles of ``min_bpm``).
    """
    _check_sampling_frequency(fs)
    if not min_bpm < max_bpm:
        raise ValueError(f"min_bpm ({min_bpm}) must be below max_bpm ({max_bpm}).")
    x = np.asarray(global_signal, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Global signal contains NaN or inf.")

    min_hz = min_bpm / SECONDS_PER_MINUTE
    max_hz = max_bpm / SECONDS_PER_MINUTE
    segment = min(int(window_seconds * fs), len(x))
    if segment < 2 * fs / min_hz:
        # Shorter than two cycles of the slowest rate searched (20 s at 6/min).
        raise ValueError(
            f"Signal too short for the respiratory rate: {segment / fs:.1f} s, "
            f"need at least {2 / min_hz:.0f} s."
        )

    # Arguments written out to match eitprocessing RateDetection (most are also the
    # scipy defaults).
    frequencies, power = signal.welch(
        x,
        fs,
        window=signal.windows.hann(segment, sym=False),
        nperseg=segment,
        noverlap=int(segment * 0.5),
        detrend="constant",
    )

    in_range = (frequencies >= min_hz) & (frequencies <= max_hz)
    range_frequencies, range_power = frequencies[in_range], power[in_range]
    if not np.any(range_power > 0):
        raise ValueError("Flat signal: no power in the respiratory range.")
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


def mdn_filter(
    signals: np.ndarray,
    fs: float,
    respiratory_rate_hz: float,
    heart_rate_hz: float,
    notch_half_width_hz: float,
    upper_limit_hz: float,
    order: int,
) -> np.ndarray:
    """Remove the cardiac component with a multiple digital notch (MDN) filter.

    Wisse et al., "Improved filtering methods to suppress cardiovascular
    contamination in EIT recordings", Physiol Meas 2024, DOI: 10.1088/1361-6579/ad46e3.
    Implementation follows eitprocessing ``MDNFilter`` (v1.8.8):

    1. one band-stop (Butterworth, zero phase) around each cardiac harmonic
       ``k * heart_rate ± notch_half_width``, up to ``upper_limit_hz``;
    2. the first band-stop never starts below (respiratory_rate + heart_rate) / 2,
       so that it does not cut the breathing;
    3. a low-pass at ``upper_limit_hz`` removes what is left above it.

    Zero-phase filtering (forward and backward) doubles the effective order of each
    filter. Apply it to the whole recording, not to short pieces: narrow band-stops
    of this order have transients of several seconds at both ends.

    The heart rate comes from the user (section 5c of the evidence notebook). It is a
    snapshot: a heart rate drifting more than about ±10/min during the recording, or
    an arrhythmia such as atrial fibrillation, is not removed by fixed notches. A heart
    rate above ``upper_limit_hz + notch_half_width_hz`` (about 230/min) gets no notch,
    only the low-pass.

    Args:
        signals:             Array with time on axis 0 (pixels or a 1-D signal).
        fs:                  Sampling frequency in Hz.
        respiratory_rate_hz: Respiratory rate in Hz.
        heart_rate_hz:       Heart rate in Hz.
        notch_half_width_hz: Half width of each band-stop, in Hz.
        upper_limit_hz:      Highest frequency filtered, in Hz.
        order:               Butterworth order of each band-stop and of the low-pass.

    Returns:
        Filtered array of the same shape as ``signals``.

    Raises:
        ValueError: if a rate is not positive, the heart rate is not above the
            respiratory rate, ``fs`` is too low for ``upper_limit_hz``, the signal is
            too short for the filter, or it has NaN at some frames only.
    """
    _check_sampling_frequency(fs)
    if respiratory_rate_hz <= 0 or heart_rate_hz <= 0:
        raise ValueError("Respiratory and heart rate must be positive (in Hz).")
    if heart_rate_hz <= respiratory_rate_hz:
        raise ValueError(
            f"Heart rate ({heart_rate_hz:.2f} Hz) must be above the respiratory rate "
            f"({respiratory_rate_hz:.2f} Hz)."
        )
    if upper_limit_hz + notch_half_width_hz >= fs / 2:
        raise ValueError(
            f"Sampling frequency {fs} Hz is too low for an upper limit of "
            f"{upper_limit_hz:.2f} Hz (Nyquist {fs / 2:.2f} Hz)."
        )

    filtered = np.asarray(signals, dtype=float)
    _check_no_gaps_in_time(filtered)
    n_harmonics = int((upper_limit_hz + notch_half_width_hz) // heart_rate_hz)

    band_stops = []
    for harmonic in range(1, n_harmonics + 1):
        centre = harmonic * heart_rate_hz
        low, high = centre - notch_half_width_hz, centre + notch_half_width_hz
        if harmonic == 1:
            # Keep the band-stop away from the breathing.
            low = max(low, (respiratory_rate_hz + heart_rate_hz) / 2)
        band_stops.append(
            signal.butter(order, [low, high], btype="bandstop", fs=fs, output="sos")
        )
    low_pass = signal.butter(order, upper_limit_hz, btype="low", fs=fs, output="sos")

    # sosfiltfilt needs more samples than its padding.
    longest_padding = max(_sosfiltfilt_padding(sos) for sos in [*band_stops, low_pass])
    if filtered.shape[0] <= longest_padding:
        raise ValueError(
            f"Signal too short for the MDN filter: {filtered.shape[0]} samples, "
            f"need more than {longest_padding}."
        )

    for band_stop in band_stops:
        filtered = signal.sosfiltfilt(band_stop, filtered, axis=0)
    return signal.sosfiltfilt(low_pass, filtered, axis=0)


def heart_rate_warnings(
    heart_rate_bpm: float,
    respiratory_rate_bpm: float,
    low_bpm: float,
    high_bpm: float,
    margin_bpm: float,
) -> list[str]:
    """Warnings about a heart rate entered by the user, before the MDN filter.

    The heart rate is not checked against the data (it is never estimated, section
    5c of the evidence notebook), so only two plausibility checks are made:
    - outside ``low_bpm``-``high_bpm`` (adult range of eitprocessing
      ``RateDetection``), e.g. a typing error;
    - within ``margin_bpm`` of 1, 2 or 3 times the respiratory rate: the cardiac
      notches then fall on the breathing or its harmonics and can change the shape
      of the breaths (section 5b).

    Args:
        heart_rate_bpm:       Heart rate given by the user, beats/min.
        respiratory_rate_bpm: Respiratory rate from the spectrum, breaths/min.
        low_bpm, high_bpm:    Plausible heart rate range.
        margin_bpm:           Distance from a multiple of the respiratory rate below
                              which a warning is given.

    Returns:
        List of warning messages, empty if none.
    """
    warnings = []
    if not low_bpm <= heart_rate_bpm <= high_bpm:
        warnings.append(
            f"Heart rate {heart_rate_bpm:.0f}/min is outside {low_bpm:.0f}-"
            f"{high_bpm:.0f}/min: check the value entered."
        )
    for multiple in (1, 2, 3):
        if abs(heart_rate_bpm - multiple * respiratory_rate_bpm) < margin_bpm:
            warnings.append(
                f"Heart rate {heart_rate_bpm:.0f}/min is close to {multiple} x the "
                f"respiratory rate ({respiratory_rate_bpm:.0f}/min): the MDN filter "
                "can change the shape of the breaths."
            )
    return warnings
