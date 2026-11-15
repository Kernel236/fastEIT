"""Functional lung mask with the watershed method.

Reimplementation of eitprocessing ``WatershedLungspace`` (v1.8.8; Somhorst et al.,
JOSS 2026;11(117):8179, DOI: 10.21105/joss.08179), with the pixel amplitude in its
"phase shift" mode (``PixelBreath``) and the small-region filter ``FilterROIBySize``.
Checked against eitprocessing on 11 real recordings
(notebooks/05_preprocessing_choices.ipynb, sections 7b, 7c and 9).

One change from eitprocessing (section 7c): the timing of each pixel is found on the
breaths used for the mask only, without their EELI. eitprocessing uses the whole
recording with a linear trend removed, so a change of EELI (a new PEEP, a
recalibration) or an artifact, even in excluded breaths, can hide most of the lung.

The mask is the VENTILATED lung, not the anatomical lung: a collapsed region with a
TIV below the threshold is outside it.

Steps:
1. TIV map: pixel value at the end of inspiration of the global signal minus the
   value at its start, mean over the breaths;
2. amplitude map: the same, but on each pixel's own timing, found by
   cross-correlation with the global signal (it catches pixels that inflate late,
   e.g. pendelluft), with the pixel and the global signal both without their EELI;
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

from fasteit.models.eit_recording import EITRecording

# Constants of eitprocessing PixelBreath (v1.8.8), not meant to be tuned.
MAX_LAG_BREATHS = 1  # pixel delay searched within ± one mean breath duration
MAX_SKIPPED_FRACTION = 0.25  # a pixel losing more breaths than this is dropped
# Islands of the mask smaller than this are removed (eitprocessing ``FilterROIBySize``
# default; notebooks/05_preprocessing_choices.ipynb, section 9).
MIN_REGION_PIXELS = 10


def tiv_per_breath(pixels: np.ndarray, breaths: np.ndarray) -> np.ndarray:
    """Tidal impedance variation of each pixel at the timing of the global signal.

    Args:
        pixels:  Array (N, rows, cols).
        breaths: Integer array (n_breaths, 3): start, peak, end frame.

    Returns:
        Array (n_breaths, rows, cols): pixel at the peak minus pixel at the start.
    """
    return pixels[breaths[:, 1]] - pixels[breaths[:, 0]]


def breath_tiv_map(recording: EITRecording, breath_id: int) -> np.ndarray:
    """TIV image of one breath, e.g. to see whether that breath is an artefact.

    Each pixel at the end of inspiration of the global signal minus its value at the
    start (``tiv_per_breath``), on the pixels of the recording (after the cardiac
    step). ``recording.tiv_map`` is the mean of these images over the valid breaths,
    so the two can be compared.

    Args:
        recording: Output of ``preprocess()``.
        breath_id: The breath, as in ``recording.breaths.breath_id``.

    Returns:
        Array (rows, cols); NaN for the pixels without a signal.

    Raises:
        ValueError: if the recording has no breath with this ``breath_id``.
    """
    row = recording.breaths[recording.breaths.breath_id == breath_id]
    if row.empty:
        raise ValueError(f"No breath {breath_id} in this recording.")
    frames = row[["start_idx", "peak_idx", "end_idx"]].to_numpy()
    return tiv_per_breath(recording.pixels, frames)[0]


def pixel_lag(
    global_detrended: np.ndarray, pixel_detrended: np.ndarray, max_lag: float
) -> int | None:
    """Lag between the global signal and a pixel, in samples.

    The cross-correlation peak closest to zero lag within ``± max_lag`` (the higher
    one if two are equally close), as in eitprocessing ``PixelBreath``. The lag is
    minus the delay of the pixel: a pixel that inflates 10 samples after the global
    signal gives -10, and its own peaks are at ``global_peaks - lag``.

    Args:
        global_detrended: Global signal without its linear trend.
        pixel_detrended:  Pixel signal without its linear trend.
        max_lag:          Largest lag searched, in samples.

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
    candidates = lags[in_range][peaks]  # lags of the cross-correlation peaks
    closest = candidates[np.abs(candidates) == np.abs(candidates).min()]
    if len(closest) == 1:
        return int(closest[0])
    # Two peaks equally close to zero lag (one on each side): the higher one.
    heights = xcorr[np.searchsorted(lags, closest)]
    return int(closest[np.argmax(heights)])


def _extremes(pixel: np.ndarray, edges: np.ndarray, find) -> np.ndarray:
    """Index of the minimum or maximum (``find``) of ``pixel`` between consecutive edges.

    An empty window (two equal edges, e.g. both clipped to the end of the recording)
    gives its edge; that breath is then skipped because its start, peak and end are
    not in order. eitprocessing crashes in this case.
    """
    positions = []
    for window_start, window_end in itertools.pairwise(edges):
        if window_end > window_start:
            positions.append(window_start + find(pixel[window_start:window_end]))
        else:
            positions.append(window_start)  # empty window
    return np.array(positions, dtype=int)


def breaths_only(x: np.ndarray, edges: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """The breaths of a signal for the cross-correlation: no EELI, nothing else.

    The EELI is the line through the signal at the breath edges (start of every
    breath and end of the last one). Without it every breath starts and ends near
    zero, so the frames outside ``frames`` can be set to zero without a step. The
    mean of ``frames`` is removed, as the linear detrend of eitprocessing does.

    Args:
        x:      1-D signal.
        edges:  Increasing frames of the breath edges.
        frames: Boolean array, True for the frames of the breaths used.

    Returns:
        Float array like ``x``, zero outside ``frames``.
    """
    tidal = x - np.interp(np.arange(len(x)), edges, x[edges])
    tidal -= tidal[frames].mean()
    tidal[~frames] = 0.0
    return tidal


def amplitude_per_breath(
    pixels: np.ndarray,
    global_signal: np.ndarray,
    breaths: np.ndarray,
    use_breath: np.ndarray | None = None,
) -> np.ndarray:
    """Amplitude of each pixel in each breath, on the pixel's own timing.

    For each pixel: the global peaks are moved to the pixel's own timing
    (``pixel_lag``, on the breaths in ``use_breath`` without their EELI,
    ``breaths_only``); between two shifted peaks the pixel minimum is its start of
    inspiration, between two minima its maximum is its peak; amplitude = peak -
    start. As in eitprocessing, the first and last breath have no value, a breath
    whose shifted peak falls outside the recording or whose order is wrong is
    skipped, and a pixel that loses more than ``MAX_SKIPPED_FRACTION`` of its
    breaths (here: of those in ``use_breath``) is left out.

    Args:
        pixels:        Array (N, rows, cols); a pixel is either NaN in every frame
                       or in none.
        global_signal: 1-D signal used for the breath detection.
        breaths:       Integer array (n_breaths, 3), consecutive breaths.
        use_breath:    Boolean array (n_breaths,): breaths the timing is found on
                       (e.g. the valid breaths); None = every breath.

    Returns:
        Array (n_breaths, rows, cols), NaN where there is no amplitude.
    """
    n_frames, n_rows, n_cols = pixels.shape
    global_peaks = breaths[:, 1]
    amplitudes = np.full((len(global_peaks), n_rows, n_cols), np.nan)
    use = np.ones(len(breaths), dtype=bool)
    if use_breath is not None:
        use = np.asarray(use_breath, dtype=bool)
    n_used = int(np.sum(use[1:-1]))  # the first and last breath never have a value
    if n_used == 0:
        return amplitudes

    edges = np.append(breaths[:, 0], breaths[-1, 2])
    frames = np.zeros(n_frames, dtype=bool)
    for start, _, end in breaths[use]:
        frames[start : end + 1] = True
    global_breaths = breaths_only(global_signal, edges, frames)
    max_lag = MAX_LAG_BREATHS * np.mean(np.diff(global_peaks))
    # Index k of ``skipped`` is the amplitude of breath k + 1 (there is none for the
    # first breath): counted if that breath is used; a k past the last breath always.
    counted = np.append(use[1:], True)
    for row in range(n_rows):
        for col in range(n_cols):
            pixel = pixels[:, row, col]
            if np.isnan(pixel).all() or np.std(pixel) == 0:
                continue
            lag = pixel_lag(global_breaths, breaths_only(pixel, edges, frames), max_lag)
            if lag is None:
                continue

            shifted = global_peaks - lag
            skipped = set(np.flatnonzero((shifted < 0) | (shifted >= n_frames)))
            shifted = np.clip(shifted, 0, n_frames - 1)
            # Minima and maxima only inside the breaths used: the window of the first
            # breath after an excluded part starts in that part. A window without any
            # frame of those breaths gives its edge, and its breath is skipped.
            minima = _extremes(np.where(frames, pixel, np.inf), shifted, np.argmin)
            starts, ends = minima[:-1], minima[1:]
            tops = _extremes(np.where(frames, pixel, -np.inf), minima, np.argmax)
            wrong_order = (starts >= tops) | (tops >= ends)
            outside = ~frames[starts] | ~frames[tops]
            skipped |= set(np.flatnonzero(wrong_order | outside))
            # As in eitprocessing, a pixel that loses every breath is not dropped
            # here: it stays all NaN. eitprocessing counts twice a breath skipped for
            # two reasons; here it counts once, which can only matter near the limit.
            lost = sum(counted[breath] for breath in skipped)
            if n_used > lost > n_used * MAX_SKIPPED_FRACTION:
                continue

            amplitude = pixel[tops] - pixel[starts]
            skipped_breaths = [breath for breath in skipped if breath < len(amplitude)]
            amplitude[skipped_breaths] = np.nan
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
    have both a TIV and an amplitude (e.g. the valid breaths); the pixel timing is
    found on the breaths in ``use_breath`` only (``amplitude_per_breath``). With every
    breath used, on a recording without changes of EELI, the masks are those of
    eitprocessing ``WatershedLungspace`` (Dice >= 0.99 on 11 recordings).

    Args:
        pixels:             Array (N, rows, cols), after the cardiac filter.
        global_signal:      1-D signal used for the breath detection.
        breaths:            Integer array (n_breaths, 3) from ``detect_breaths``.
        use_breath:         Boolean array (n_breaths,): breaths to average over.
        threshold_fraction: Fraction of the maximum TIV (0.15 in eitprocessing).

    Returns:
        ``mask`` (rows, cols) bool, mean ``tiv_map`` and mean ``amplitude_map``. The
        first and last breath are never in the means (they have no amplitude); a
        pixel without data is NaN in the maps.

    Raises:
        ValueError: if no breath can be used, the TIV map has no positive maximum
            (inverted signal), or no amplitude peak lies inside the TIV mask.
    """
    tiv = tiv_per_breath(pixels, breaths)
    amplitude = amplitude_per_breath(pixels, global_signal, breaths, use_breath)
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
    if not threshold > 0:
        raise ValueError(
            "The TIV map has no positive maximum: check that impedance rises with "
            "inspiration."
        )
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


def remove_small_regions(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Remove the islands of the mask smaller than ``min_pixels`` (step 5).

    Pixels belong to the same island when they share a side (4-connectivity), as in
    eitprocessing ``FilterROIBySize``.

    Args:
        mask: Boolean mask, shape (32, 32).
        min_pixels: Smallest island kept.

    Returns:
        The mask without the smaller islands.
    """
    side_neighbours = ndimage.generate_binary_structure(2, 1)
    labels, n_regions = ndimage.label(mask, structure=side_neighbours)
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n_regions + 1))
    return np.isin(labels, np.flatnonzero(sizes >= min_pixels) + 1)
