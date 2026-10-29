"""Tests for the breath detection module."""

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.breaths import (
    detect_breaths,
    drop_doubles,
    drop_low_amplitude,
    fill_with_nearest,
    invalid_samples,
    local_baseline,
    peaks_and_valleys,
    trim_edges,
)

CFG = PreprocessingConfig()


def breathing(rate_hz=0.5, fs=50, seconds=60, amplitude=1.0):
    t = np.arange(0, seconds, 1 / fs)
    return amplitude * np.sin(2 * np.pi * rate_hz * t)


# ── invalid data ───────────────────────────────────────────────────────────────


def invalid(x):
    return invalid_samples(x, CFG.invalid_data_percentile, CFG.invalid_data_multiplier)


def test_clean_breathing_has_no_invalid_samples():
    assert len(invalid(breathing())) == 0


def test_a_disconnection_spike_is_invalid():
    x = breathing()
    x[1000:1005] = 50.0
    assert list(invalid(x)) == [1000, 1001, 1002, 1003, 1004]


def test_fill_with_nearest_splits_a_run_between_its_neighbours():
    x = np.array([1.0, np.nan, np.nan, np.nan, np.nan, 3.0])
    filled = fill_with_nearest(x, np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(filled, [1, 1, 1, 3, 3, 3])


def test_fill_with_nearest_at_the_edges():
    x = np.array([np.nan, np.nan, 2.0, 5.0, np.nan])
    filled = fill_with_nearest(x, np.array([0, 1, 4]))
    np.testing.assert_array_equal(filled, [2, 2, 2, 5, 5])


def test_fill_with_nearest_does_not_change_the_input():
    x = np.array([1.0, 9.0, 3.0])
    fill_with_nearest(x, np.array([1]))
    assert x[1] == 9.0


def test_fill_with_nearest_rejects_an_all_missing_signal():
    with pytest.raises(ValueError, match="Every sample"):
        fill_with_nearest(np.zeros(3), np.array([0, 1, 2]))


# ── baseline, peaks and valleys ─────────────────────────────────────────────────


def candidates(x, fs=50):
    base = local_baseline(x, fs, CFG.breath_baseline_seconds)
    return peaks_and_valleys(x, base, fs, CFG.breath_min_duration_seconds)


def test_baseline_of_a_constant_signal_is_the_constant():
    np.testing.assert_allclose(local_baseline(np.full(2000, 7.0), 50, 15), 7.0)


def test_baseline_follows_a_step_of_eeli():
    x = np.r_[np.zeros(3000), np.ones(3000)]  # e.g. a PEEP step
    base = local_baseline(x, fs=50, seconds=15)
    assert base[100] == pytest.approx(0) and base[-100] == pytest.approx(1)


def test_baseline_on_a_signal_shorter_than_the_window():
    assert local_baseline(np.arange(10.0), fs=50, seconds=15).shape == (10,)


def test_one_peak_and_one_valley_per_breath():
    peaks, valleys = candidates(breathing(rate_hz=0.5, seconds=60))  # 30 breaths
    assert 29 <= len(peaks) <= 30 and 29 <= len(valleys) <= 30


def test_breaths_are_found_on_both_sides_of_a_peep_step():
    x = breathing(seconds=120) + np.r_[np.zeros(3000), 5 * np.ones(3000)]
    peaks, _ = candidates(x)
    assert np.sum(peaks < 3000) >= 25 and np.sum(peaks > 3000) >= 25


def test_ripples_closer_than_the_minimum_duration_are_not_extra_peaks():
    x = breathing(seconds=60) + 0.05 * breathing(rate_hz=5, seconds=60)
    peaks, _ = candidates(x)
    assert len(peaks) <= 31


# ── edges and doubles ──────────────────────────────────────────────────────────


def test_two_peaks_between_two_valleys_keep_the_higher():
    signal = np.array([0, 5, 2, 8, 0], dtype=float)
    peaks, valleys = drop_doubles(signal, np.array([1, 3]), np.array([0, 4]))
    assert list(peaks) == [3] and list(valleys) == [0, 4]


def test_two_valleys_without_a_peak_keep_the_lower():
    signal = np.array([1, 0, 9, 0.5, -1, 9, 0], dtype=float)
    peaks, valleys = drop_doubles(signal, np.array([2, 5]), np.array([1, 3, 4, 6]))
    assert list(valleys) == [1, 4, 6] and list(peaks) == [2, 5]


def test_peaks_and_valleys_alternate_after_drop_doubles():
    x = breathing(seconds=60) + 0.3 * breathing(rate_hz=1.3, seconds=60)
    base = local_baseline(x, 50, CFG.breath_baseline_seconds)
    peaks, valleys = peaks_and_valleys(x, base, 50, CFG.breath_min_duration_seconds)
    peaks, valleys = drop_doubles(x, *trim_edges(x, peaks, valleys, base))
    assert len(peaks) == len(valleys) - 1
    assert np.all((valleys[:-1] < peaks) & (peaks < valleys[1:]))


def test_trim_edges_keeps_only_peaks_between_the_first_and_last_valley():
    x = breathing(seconds=20)
    base = local_baseline(x, 50, CFG.breath_baseline_seconds)
    peaks, valleys = peaks_and_valleys(x, base, 50, CFG.breath_min_duration_seconds)
    peaks, valleys = trim_edges(x, peaks, valleys, base)
    assert peaks.min() > valleys.min() and peaks.max() < valleys.max()


def test_trim_edges_without_valleys():
    peaks, valleys = trim_edges(
        np.zeros(10), np.array([3]), np.array([], int), np.zeros(10)
    )
    assert len(valleys) == 0 and list(peaks) == [3]


# ── low amplitude and full detection ───────────────────────────────────────────


def detect(x, fs=50):
    return detect_breaths(
        x,
        fs,
        invalid_percentile=CFG.invalid_data_percentile,
        invalid_multiplier=CFG.invalid_data_multiplier,
        baseline_seconds=CFG.breath_baseline_seconds,
        min_duration_seconds=CFG.breath_min_duration_seconds,
        low_amplitude_fraction=CFG.breath_low_amplitude_fraction,
        invalid_margin_seconds=CFG.invalid_data_margin_seconds,
    )


def test_a_small_peak_is_merged_into_the_breath_around_it():
    signal = np.array([0, 10, 0, 10, 0, 1, 0.5, 10, 0], dtype=float)
    peaks, valleys = drop_low_amplitude(
        signal, np.array([1, 3, 5, 7]), np.array([0, 2, 4, 6, 8]), 0.25
    )
    assert list(peaks) == [1, 3, 7]
    assert list(valleys) == [0, 2, 4, 8]


def test_breaths_of_a_regular_signal():
    breaths, near_invalid = detect(breathing(rate_hz=0.5, seconds=60))  # 30 breaths
    assert 27 <= len(breaths) <= 29  # incomplete breaths at the edges are dropped
    assert np.all(breaths[:, 0] < breaths[:, 1]) and np.all(
        breaths[:, 1] < breaths[:, 2]
    )
    assert np.all(
        breaths[1:, 0] == breaths[:-1, 2]
    )  # each breath starts where the last ended
    assert not near_invalid.any()


def test_breaths_near_a_spike_are_flagged_not_removed():
    clean, _ = detect(breathing(seconds=60))
    x = breathing(seconds=60)
    x[1500:1503] = 100.0
    breaths, near_invalid = detect(x)
    assert len(breaths) == len(clean)
    assert 1 <= near_invalid.sum() <= 3


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(3000),
        np.full(3000, 0.1),  # the mean of 0.1 repeated is not exactly 0.1
        breathing(seconds=1),
        np.array([1.0]),
        np.array([1.0, 2.0]),
    ],
)
def test_no_breaths_in_a_flat_or_too_short_signal(x):
    breaths, near_invalid = detect(x)
    assert breaths.shape == (0, 3) and near_invalid.shape == (0,)


@pytest.mark.parametrize(
    "x, fs, message",
    [
        (np.array([1.0, np.nan, 2.0]), 50, "NaN"),
        (np.array([]), 50, "non-empty 1-D"),
        (np.zeros((100, 2)), 50, "non-empty 1-D"),
        (breathing(), 0, "positive"),
    ],
)
def test_detect_breaths_rejects_unusable_input(x, fs, message):
    with pytest.raises(ValueError, match=message):
        detect(x, fs)
