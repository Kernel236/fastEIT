"""Tests for the lung mask module."""

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.breaths import detect_breaths
from fasteit.preprocessing.lung_mask import (
    amplitude_per_breath,
    pixel_lag,
    remove_small_regions,
    tiv_per_breath,
    watershed_mask,
)

CFG = PreprocessingConfig()
FS = 20


def breathing(t, delay_s=0.0, rate_hz=0.25):
    """Inflation from 0 to 1 and back, 15 breaths/min, delayed by ``delay_s``."""
    return np.sin(np.pi * rate_hz * (t - delay_s)) ** 2


def synthetic_recording():
    """16 x 16 image: one lung whose dorsal part inflates 1 s late, and an artifact.

    The artifact oscillates in antiphase (2 s late) on the right of the image.
    """
    t = np.arange(0, 90, 1 / FS)
    rows, cols = np.mgrid[0:16, 0:16]
    lung = np.exp(-(((rows - 6) / 4) ** 2 + ((cols - 5) / 3) ** 2))
    late = (rows >= 8) & (lung > 0.1)
    artifact = 0.8 * np.exp(-(((rows - 6) / 1.2) ** 2 + ((cols - 13) / 1.2) ** 2))

    pixels = breathing(t)[:, None, None] * np.where(late, 0, lung)
    pixels += breathing(t, 1.0)[:, None, None] * np.where(late, lung, 0)
    pixels += breathing(t, 2.0)[:, None, None] * artifact
    pixels += 0.01 * np.random.default_rng(0).normal(size=pixels.shape)

    global_signal = pixels.sum(axis=(1, 2))
    breaths, _ = detect_breaths(
        global_signal,
        FS,
        invalid_percentile=CFG.invalid_data_percentile,
        invalid_multiplier=CFG.invalid_data_multiplier,
        baseline_seconds=CFG.breath_baseline_seconds,
        min_duration_seconds=CFG.breath_min_duration_seconds,
        low_amplitude_fraction=CFG.breath_low_amplitude_fraction,
        invalid_margin_seconds=CFG.invalid_data_margin_seconds,
    )
    is_lung = lung >= CFG.lung_mask_threshold_fraction
    return pixels, global_signal, breaths, is_lung & late, artifact > 0.15


def mask_of(pixels, global_signal, breaths, use_breath=None):
    if use_breath is None:
        use_breath = np.ones(len(breaths), dtype=bool)
    return watershed_mask(
        pixels, global_signal, breaths, use_breath, CFG.lung_mask_threshold_fraction
    )


# --- maps ---


def test_tiv_per_breath_is_peak_minus_start_of_the_global_breath():
    pixels = np.arange(10.0)[:, None, None] * np.ones((1, 2, 2))
    tiv = tiv_per_breath(pixels, np.array([[1, 4, 6], [6, 9, 9]]))
    np.testing.assert_array_equal(tiv[:, 0, 0], [3.0, 3.0])


def test_pixel_lag_finds_the_delay():
    t = np.arange(0, 60, 1 / FS)
    reference = breathing(t) - breathing(t).mean()
    late = breathing(t, 0.5) - breathing(t, 0.5).mean()
    assert pixel_lag(reference, late, max_lag=4 * FS) == pytest.approx(-0.5 * FS, abs=1)


def test_pixel_lag_of_a_flat_correlation_is_none():
    assert pixel_lag(np.zeros(100), np.zeros(100), max_lag=20) is None


def test_amplitude_has_no_value_for_the_first_and_last_breath():
    pixels, global_signal, breaths, _, _ = synthetic_recording()
    amplitude = amplitude_per_breath(pixels, global_signal, breaths)
    assert np.isnan(amplitude[[0, -1]]).all()
    assert not np.isnan(amplitude[1:-1, 6, 5]).any()


def test_amplitude_needs_at_least_three_breaths():
    pixels, global_signal, breaths, _, _ = synthetic_recording()
    assert np.isnan(amplitude_per_breath(pixels, global_signal, breaths[:2])).all()


# --- watershed ---


def test_watershed_keeps_the_late_lung_and_drops_the_artifact():
    pixels, global_signal, breaths, late_lung, artifact = synthetic_recording()
    mask, tiv_map, _ = mask_of(pixels, global_signal, breaths)
    tiv_mask = tiv_map >= CFG.lung_mask_threshold_fraction * tiv_map.max()
    assert mask[late_lung].all()
    assert not tiv_mask[late_lung].all()  # the TIV map alone loses the late part
    assert not mask[artifact].any()


def test_maps_are_averaged_only_over_the_chosen_breaths():
    pixels, global_signal, breaths, _, _ = synthetic_recording()
    pixels = pixels.copy()
    pixels[breaths[5, 0] : breaths[5, 2]] *= 3  # one much larger breath
    use = np.ones(len(breaths), dtype=bool)
    use[5] = False
    _, with_it, _ = mask_of(pixels, global_signal, breaths)
    _, without_it, _ = mask_of(pixels, global_signal, breaths, use)
    assert without_it.max() < with_it.max()


def test_no_usable_breath_is_an_error():
    pixels, global_signal, breaths, _, _ = synthetic_recording()
    with pytest.raises(ValueError, match="No usable breaths"):
        mask_of(pixels, global_signal, breaths, np.zeros(len(breaths), dtype=bool))


# --- small regions ---


def test_islands_smaller_than_the_minimum_are_removed():
    mask = np.zeros((12, 12), dtype=bool)
    mask[0:2, 0:5] = True  # 10 pixels: kept
    mask[6:9, 6:9] = True  # 9 pixels: removed
    cleaned = remove_small_regions(mask, CFG.lung_mask_min_region_pixels)
    assert cleaned[0:2, 0:5].all() and not cleaned[6:9, 6:9].any()


def test_pixels_touching_only_by_a_corner_are_separate_islands():
    mask = np.zeros((12, 12), dtype=bool)
    mask[0:3, 0:3] = True  # 9 pixels
    mask[3:6, 3:6] = True  # 9 pixels, touching the first by one corner
    assert not remove_small_regions(mask, CFG.lung_mask_min_region_pixels).any()


# --- edge cases ---


def test_amplitude_of_a_pixel_is_its_tidal_swing():
    pixels, global_signal, breaths, _, _ = synthetic_recording()
    amplitude = amplitude_per_breath(pixels, global_signal, breaths)
    lung_centre = np.exp(0.0)  # the lung peak is 1 at row 6, column 5
    np.testing.assert_allclose(amplitude[1:-1, 6, 5], lung_centre, atol=0.05)


def test_peaks_moved_out_of_the_recording_do_not_crash():
    # Global peaks 0 and 5 move before the start for a pixel 30 samples late:
    # both windows are clipped to frame 0 and become empty.
    n_frames = 400
    t = np.arange(n_frames)
    pixel = np.sin(2 * np.pi * (t - 30) / 80) ** 2
    pixels = pixel[:, None, None] * np.ones((1, 1, 2))
    global_signal = np.sin(2 * np.pi * t / 80) ** 2
    peaks = np.array([0, 5, 60, 140, 220, 300, 380])
    breaths = np.column_stack([peaks - 1, peaks, peaks + 1])
    amplitude = amplitude_per_breath(pixels, global_signal, breaths)
    assert amplitude.shape == (7, 1, 2)


def test_an_inverted_signal_has_no_lung_mask():
    # Every pixel falls while the global signal rises: the TIV map is all negative.
    pixels, global_signal, breaths, _, _ = synthetic_recording()
    falling = -global_signal[:, None, None] * np.ones((1, 3, 3))
    with pytest.raises(ValueError, match="no positive maximum"):
        mask_of(falling, global_signal, breaths)
