"""Tests for Dräger .bin utility functions (sentinel handling, fs estimation, slicing)."""

import numpy as np
import pytest

from fasteit.parsers.draeger.bin.bin_utils import (
    _BIT_SENTINELS,
    _FLOAT_SENTINELS,
    estimate_sampling_frequency_hz,
    is_not_connected_sentinel,
    normalize_frame_slice,
    replace_no_data_sentinels,
)
from fasteit.parsers.errors import InvalidSliceError

# ── normalize_frame_slice ──────────────────────────────────────────────────────


def test_normalize_default_slice():
    start, stop = normalize_frame_slice(0, None, 100)
    assert start == 0
    assert stop == 100


def test_normalize_with_max_frames():
    start, stop = normalize_frame_slice(10, 50, 100)
    assert start == 10
    assert stop == 60


def test_normalize_clamps_stop_to_total():
    start, stop = normalize_frame_slice(90, 50, 100)
    assert stop == 100


def test_normalize_negative_first_frame_raises():
    with pytest.raises(InvalidSliceError):
        normalize_frame_slice(-1, None, 100)


def test_normalize_zero_max_frames_raises():
    with pytest.raises(InvalidSliceError):
        normalize_frame_slice(0, 0, 100)


def test_normalize_first_frame_out_of_bounds_raises():
    with pytest.raises(InvalidSliceError):
        normalize_frame_slice(100, None, 100)


# ── is_not_connected_sentinel ─────────────────────────────────────────────────


def test_sentinel_detection_identifies_bit_pattern():
    # Build a float32 array with the sentinel bit pattern embedded
    sentinel_int = 0xFF7FC99E
    arr = np.zeros(4, dtype=np.float32)
    arr.view(np.uint32)[2] = sentinel_int  # plant sentinel at index 2
    mask = is_not_connected_sentinel(arr, (sentinel_int,))
    assert mask[2] is np.bool_(True)
    assert not mask[0]
    assert not mask[1]


def test_sentinel_detection_no_sentinels_returns_false():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mask = is_not_connected_sentinel(arr, ())
    assert not mask.any()


# ── replace_no_data_sentinels ─────────────────────────────────────────────────


def test_replace_float_sentinel_with_nan():
    arr = np.array([-1000.0, 1.5, -1000.0, 2.0], dtype=np.float32)
    result = replace_no_data_sentinels(arr, _FLOAT_SENTINELS, ())
    assert np.isnan(result[0])
    assert np.isnan(result[2])
    assert result[1] == pytest.approx(1.5)


def test_replace_bit_sentinel_with_nan():
    arr = np.zeros(3, dtype=np.float32)
    arr.view(np.uint32)[1] = _BIT_SENTINELS[0]
    result = replace_no_data_sentinels(arr, (), _BIT_SENTINELS)
    assert np.isnan(result[1])
    assert not np.isnan(result[0])
    assert not np.isnan(result[2])


def test_replace_does_not_modify_original():
    arr = np.array([-1000.0, 1.0], dtype=np.float32)
    _ = replace_no_data_sentinels(arr, _FLOAT_SENTINELS, ())
    assert arr[0] == pytest.approx(-1000.0)  # original unchanged


# ── estimate_sampling_frequency_hz ───────────────────────────────────────────


def test_fs_estimation_50hz():
    # 50 frames at 50 Hz → dt = 1/50 s = 0.00002 day fractions per frame
    dt_day = 1.0 / (50.0 * 86400.0)
    ts = np.arange(50) * dt_day
    fs = estimate_sampling_frequency_hz(ts)
    assert abs(fs - 50.0) < 0.1


def test_fs_estimation_20hz():
    dt_day = 1.0 / (20.0 * 86400.0)
    ts = np.arange(100) * dt_day
    fs = estimate_sampling_frequency_hz(ts)
    assert abs(fs - 20.0) < 0.1


def test_fs_estimation_requires_at_least_2_timestamps():
    with pytest.raises(ValueError, match="At least 2"):
        estimate_sampling_frequency_hz(np.array([0.5]))


def test_fs_estimation_raises_on_identical_timestamps():
    with pytest.raises(ValueError, match="Non-positive"):
        estimate_sampling_frequency_hz(np.array([0.5, 0.5, 0.5]))
