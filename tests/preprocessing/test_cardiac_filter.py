"""Tests for the cardiac filter module."""

import numpy as np

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.cardiac_filter import moving_average, moving_average_window


def sine(frequency_hz, fs, seconds=60.0, amplitude=1.0):
    t = np.arange(0, seconds, 1 / fs)
    return amplitude * np.sin(2 * np.pi * frequency_hz * t)


# ── moving average ─────────────────────────────────────────────────────────────


def test_window_is_odd_and_close_to_requested_duration():
    assert moving_average_window(fs=50, seconds=0.1) == 5
    assert moving_average_window(fs=20, seconds=0.1) == 3
    assert moving_average_window(fs=50, seconds=0.0) == 1


def test_default_window_is_0_1_seconds():
    assert PreprocessingConfig().moving_average_seconds == 0.1


def test_breathing_passes_almost_untouched():
    breathing = sine(0.5, fs=50)  # 30 breaths/min
    smoothed = moving_average(breathing, fs=50)
    assert np.ptp(smoothed) / np.ptp(breathing) > 0.99


def test_fast_noise_is_removed():
    noise = sine(10.0, fs=50)  # 5-sample window: 10 Hz is cancelled
    smoothed = moving_average(noise, fs=50)
    assert np.ptp(smoothed[10:-10]) < 0.05


def test_pixels_keep_their_shape_and_are_smoothed_along_time():
    pixels = np.random.default_rng(0).normal(size=(200, 32, 32))
    smoothed = moving_average(pixels, fs=50)
    assert smoothed.shape == pixels.shape
    assert smoothed.std() < pixels.std()
