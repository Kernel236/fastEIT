"""Functional lung mask with the watershed method.

Reimplementation of eitprocessing ``WatershedLungspace`` (v1.8.8; Somhorst et al.,
JOSS 2026;11(117):8179, DOI: 10.21105/joss.08179), with the pixel amplitude in its
"phase shift" mode (``PixelBreath``) and the small-region filter ``FilterROIBySize``.
Checked to give the same masks on 11 real recordings
(notebooks/05_preprocessing_choices.ipynb, sections 7b and 9).

Steps:
1. TIV map: pixel value at the end of inspiration of the global signal minus the
   value at its start, mean over the breaths;
2. amplitude map: the same, but on each pixel's own timing, found by
   cross-correlation with the global signal (it catches pixels that inflate late,
   e.g. pendelluft);
3. threshold: a fraction of the maximum of the TIV map, applied to both maps;
4. watershed: the amplitude map is split into regions around its local maxima; a
   region is kept if its maximum is inside the TIV mask. Mask = kept regions inside
   the amplitude mask;
5. islands smaller than a minimum size are removed.

The watershed assumes smooth images, with high amplitude in the centre of the lung
and low amplitude at its edges, as reconstructed EIT images are.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import signal

# Constants of eitprocessing PixelBreath (v1.8.8), not meant to be tuned.
MAX_LAG_BREATHS = 1  # pixel delay searched within ± one mean breath duration
MAX_SKIPPED_FRACTION = 0.25  # a pixel losing more breaths than this is dropped


def tiv_per_breath(pixels: np.ndarray, breaths: np.ndarray) -> np.ndarray:
    """Tidal impedance variation of each pixel at the timing of the global signal.

    Args:
        pixels:  Array (N, rows, cols).
        breaths: Integer array (n_breaths, 3): start, peak, end frame.

    Returns:
        Array (n_breaths, rows, cols): pixel at the peak minus pixel at the start.
    """
    return pixels[breaths[:, 1]] - pixels[breaths[:, 0]]


def pixel_lag(
    global_detrended: np.ndarray, pixel_detrended: np.ndarray, max_lag: float
) -> int | None:
    """Delay of a pixel behind the global signal, in samples.

    The cross-correlation peak closest to zero lag within ``± max_lag`` (the higher
    one if two are equally close), as in eitprocessing ``PixelBreath``.

    Returns:
        The lag, or None if the cross-correlation has no clear positive peak.
    """
    n = len(global_detrended)
    xcorr = signal.correlate(global_detrended, pixel_detrended, mode="same")
    lags = signal.correlation_lags(n, n, mode="same")
    in_range = np.abs(lags) <= max_lag
    peaks, _ = signal.find_peaks(xcorr[in_range], height=0, prominence=xcorr.std())
    if len(peaks) == 0:
        return None
    candidates = lags[in_range][peaks]
    closest = candidates[np.abs(candidates) == np.abs(candidates).min()]
    if len(closest) == 1:
        return int(closest[0])
    return int(closest[np.argmax(xcorr[np.searchsorted(lags, closest)])])


def amplitude_per_breath(
    pixels: np.ndarray, global_signal: np.ndarray, breaths: np.ndarray
) -> np.ndarray:
    """Amplitude of each pixel in each breath, on the pixel's own timing.

    For each pixel: the global peaks are shifted by the pixel's delay
    (``pixel_lag``); between two shifted peaks the pixel minimum is its start of
    inspiration, between two minima its maximum is its peak; amplitude = peak -
    start. As in eitprocessing, the first and last breath have no value, a breath
    whose shifted peak falls outside the recording or whose order is wrong is
    skipped, and a pixel that loses more than ``MAX_SKIPPED_FRACTION`` of its
    breaths is left out.

    Args:
        pixels:        Array (N, rows, cols) without NaN at some frames only.
        global_signal: 1-D signal used for the breath detection.
        breaths:       Integer array (n_breaths, 3), consecutive breaths.

    Returns:
        Array (n_breaths, rows, cols), NaN where there is no amplitude.
    """
    n_frames, n_rows, n_cols = pixels.shape
    global_peaks = breaths[:, 1]
    amplitudes = np.full((len(global_peaks), n_rows, n_cols), np.nan)
    if len(global_peaks) < 3:
        return amplitudes  # first and last breath never have a value

    global_detrended = signal.detrend(global_signal, type="linear")
    max_lag = MAX_LAG_BREATHS * np.mean(np.diff(global_peaks))
    for row in range(n_rows):
        for col in range(n_cols):
            pixel = pixels[:, row, col]
            if np.isnan(pixel).all() or np.std(pixel) == 0:
                continue
            lag = pixel_lag(
                global_detrended, signal.detrend(pixel, type="linear"), max_lag
            )
            if lag is None:
                continue

            shifted = global_peaks - lag
            skipped = set(np.flatnonzero((shifted < 0) | (shifted >= n_frames)))
            shifted = np.clip(shifted, 0, n_frames - 1)
            minima = np.array(
                [a + np.argmin(pixel[a:b]) for a, b in itertools.pairwise(shifted)]
            )
            starts, ends = minima[:-1], minima[1:]
            tops = np.array(
                [a + np.argmax(pixel[a:b]) for a, b in zip(starts, ends, strict=True)],
                dtype=int,
            )
            skipped |= set(np.flatnonzero((starts >= tops) | (tops >= ends)))
            # Same condition as eitprocessing (a pixel losing every breath is kept,
            # all NaN).
            if len(starts) > len(skipped) > len(starts) * MAX_SKIPPED_FRACTION:
                continue

            amplitude = pixel[tops] - pixel[starts]
            amplitude[[i for i in skipped if i < len(amplitude)]] = np.nan
            amplitudes[1:-1, row, col] = amplitude
    return amplitudes
