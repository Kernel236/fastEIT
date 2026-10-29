"""Tests for the cardiac filter module."""

import numpy as np
import pytest
from scipy import signal as scipy_signal

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.cardiac_filter import (
    mdn_filter,
    moving_average,
    moving_average_window,
    respiratory_rate,
)


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


# ── respiratory rate ───────────────────────────────────────────────────────────


def test_respiratory_rate_of_a_30_per_minute_sine():
    breathing = sine(0.5, fs=50, seconds=120)
    assert abs(respiratory_rate(breathing, fs=50) * 60 - 30) < 0.5


def test_respiratory_rate_ignores_a_smaller_cardiac_component():
    signal_ = sine(0.25, fs=50, seconds=120) + sine(
        1.5, fs=50, seconds=120, amplitude=0.1
    )
    assert abs(respiratory_rate(signal_, fs=50) * 60 - 15) < 0.5


def test_respiratory_rate_works_on_a_signal_shorter_than_the_window():
    breathing = sine(0.5, fs=20, seconds=20)  # 20 s < 30 s Welch window
    assert abs(respiratory_rate(breathing, fs=20) * 60 - 30) < 2


# ── MDN filter ─────────────────────────────────────────────────────────────────


def power_near(x, fs, f0, half_width=0.05):
    frequencies, power = scipy_signal.welch(x, fs, nperseg=len(x) // 4)
    return power[np.abs(frequencies - f0) < half_width].sum()


def test_mdn_removes_the_heart_and_keeps_the_breathing():
    fs = 50
    breathing = sine(0.5, fs, seconds=120)
    heart = sine(1.5, fs, seconds=120, amplitude=0.2)
    filtered = mdn_filter(
        breathing + heart, fs, respiratory_rate_hz=0.5, heart_rate_hz=1.5
    )
    cardiac_drop_db = 10 * np.log10(
        power_near(filtered, fs, 1.5) / power_near(heart, fs, 1.5)
    )
    assert cardiac_drop_db < -30
    assert np.ptp(filtered[500:-500]) / np.ptp(breathing[500:-500]) > 0.99


def test_mdn_filters_pixels_along_time():
    pixels = np.random.default_rng(1).normal(size=(1000, 4, 4))
    filtered = mdn_filter(pixels, fs=50, respiratory_rate_hz=0.5, heart_rate_hz=1.5)
    assert filtered.shape == pixels.shape


def test_mdn_rejects_a_heart_rate_not_above_the_respiratory_rate():
    with pytest.raises(ValueError, match="must be above"):
        mdn_filter(np.zeros(500), fs=50, respiratory_rate_hz=0.5, heart_rate_hz=0.5)


def test_heart_rate_is_not_set_by_default():
    assert PreprocessingConfig().heart_rate_bpm is None
