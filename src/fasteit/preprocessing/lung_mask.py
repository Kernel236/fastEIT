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
import warnings

import numpy as np
from scipy import ndimage, signal
from skimage import feature, segmentation

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


def watershed_mask(
    pixels: np.ndarray,
    global_signal: np.ndarray,
    breaths: np.ndarray,
    use_breath: np.ndarray,
    threshold_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lung mask from the TIV and amplitude maps (steps 1-4).

    TIV and amplitude are computed on all breaths, because the pixel timing needs
    consecutive breaths, and averaged only over the breaths in ``use_breath`` that
    have both a TIV and an amplitude (e.g. the valid breaths). With every breath used
    the result is the same as eitprocessing ``WatershedLungspace``.

    Args:
        pixels:             Array (N, rows, cols), after the cardiac filter.
        global_signal:      1-D signal used for the breath detection.
        breaths:            Integer array (n_breaths, 3) from ``detect_breaths``.
        use_breath:         Boolean array (n_breaths,): breaths to average over.
        threshold_fraction: Fraction of the maximum TIV (0.15 in eitprocessing).

    Returns:
        ``mask`` (rows, cols) bool, mean ``tiv_map`` and mean ``amplitude_map``.

    Raises:
        ValueError: if no breath can be used or no amplitude peak lies inside the TIV
            mask.
    """
    tiv = tiv_per_breath(pixels, breaths)
    amplitude = amplitude_per_breath(pixels, global_signal, breaths)
    has_values = ~np.isnan(tiv).all(axis=(1, 2)) & ~np.isnan(amplitude).all(axis=(1, 2))
    used = has_values & np.asarray(use_breath, dtype=bool)
    if not used.any():
        raise ValueError("No usable breaths: the lung mask cannot be computed.")

    with warnings.catch_warnings():
        # A pixel without any amplitude (e.g. outside the belt) stays NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        tiv_map = np.nanmean(tiv[used], axis=0)
        amplitude_map = np.nanmean(amplitude[used], axis=0)
    threshold = threshold_fraction * np.nanmax(tiv_map)
    tiv_mask = tiv_map >= threshold
    amplitude_mask = amplitude_map >= threshold

    # One numbered marker per local maximum of the amplitude map.
    peak_rows_cols = feature.peak_local_max(
        np.nan_to_num(amplitude_map), exclude_border=False
    )
    is_peak = np.zeros(amplitude_map.shape, dtype=bool)
    is_peak[tuple(peak_rows_cols.T)] = True
    markers, _ = ndimage.label(is_peak)
    kept = np.unique(markers[is_peak & tiv_mask])
    if len(kept) == 0:
        raise ValueError("No amplitude peak inside the TIV mask: no lung mask.")

    regions = segmentation.watershed(-amplitude_map, markers)
    mask = np.isin(regions, kept) & amplitude_mask
    return mask, tiv_map, amplitude_map
