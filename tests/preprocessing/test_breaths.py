"""Tests for the breath detection module."""

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.breaths import fill_with_nearest, invalid_samples

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
