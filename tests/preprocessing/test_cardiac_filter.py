"""Tests for the cardiac filter module.

Parameters come from PreprocessingConfig, so the tests also check that the defaults
used by the pipeline give the expected behaviour.
"""

import numpy as np
import pytest
from scipy import signal as scipy_signal

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.cardiac_filter import (
    SECONDS_PER_MINUTE,
    mdn_filter,
    moving_average,
    moving_average_window,
    respiratory_rate,
)

CFG = PreprocessingConfig()


def sine(frequency_hz, fs, seconds=60.0, amplitude=1.0):
    t = np.arange(0, seconds, 1 / fs)
    return amplitude * np.sin(2 * np.pi * frequency_hz * t)


def rate(x, fs):
    return respiratory_rate(
        x,
        fs,
        min_bpm=CFG.respiratory_rate_min_bpm,
        max_bpm=CFG.respiratory_rate_max_bpm,
        window_seconds=CFG.spectrum_window_seconds,
    )


def mdn(x, fs, respiratory_hz, heart_hz):
    return mdn_filter(
        x,
        fs,
        respiratory_hz,
        heart_hz,
        notch_half_width_hz=CFG.mdn_notch_half_width_bpm / SECONDS_PER_MINUTE,
        upper_limit_hz=CFG.mdn_upper_limit_bpm / SECONDS_PER_MINUTE,
        order=CFG.mdn_order,
    )


def power_near(x, fs, f0, half_width=0.05):
    frequencies, power = scipy_signal.welch(x, fs, nperseg=len(x) // 4)
    return power[np.abs(frequencies - f0) < half_width].sum()


# ── config defaults ────────────────────────────────────────────────────────────


def test_config_defaults():
    assert CFG.moving_average_seconds == 0.1
    assert CFG.heart_rate_bpm is None
    assert (CFG.respiratory_rate_min_bpm, CFG.respiratory_rate_max_bpm) == (6.0, 60.0)


# ── moving average ─────────────────────────────────────────────────────────────


def test_window_is_odd_and_close_to_requested_duration():
    assert moving_average_window(fs=50, seconds=0.1) == 5
    assert moving_average_window(fs=20, seconds=0.1) == 3
    assert moving_average_window(fs=50, seconds=0.0) == 1


def test_window_rejects_non_positive_fs():
    with pytest.raises(ValueError, match="positive"):
        moving_average_window(fs=0, seconds=0.1)


def test_moving_average_exact_values_inside_the_signal():
    x = np.arange(20, dtype=float) ** 2
    smoothed = moving_average(x, fs=50, seconds=CFG.moving_average_seconds)
    expected = np.convolve(x, np.ones(5) / 5, mode="valid")
    np.testing.assert_allclose(smoothed[2:-2], expected)


def test_moving_average_keeps_a_constant_signal_unchanged_at_the_edges():
    x = np.full(50, 3.0)
    np.testing.assert_allclose(moving_average(x, fs=50, seconds=0.1), x)


def test_breathing_passes_almost_untouched():
    breathing = sine(0.5, fs=50)  # 30 breaths/min
    smoothed = moving_average(breathing, fs=50, seconds=CFG.moving_average_seconds)
    assert np.ptp(smoothed) / np.ptp(breathing) > 0.99


def test_fast_noise_is_removed():
    noise = sine(10.0, fs=50)  # 5-sample window: 10 Hz is cancelled
    smoothed = moving_average(noise, fs=50, seconds=CFG.moving_average_seconds)
    assert np.ptp(smoothed[10:-10]) < 0.05


def test_pixels_keep_their_shape_and_are_smoothed_along_time():
    pixels = np.random.default_rng(0).normal(size=(200, 32, 32))
    smoothed = moving_average(pixels, fs=50, seconds=0.1)
    assert smoothed.shape == pixels.shape
    assert smoothed.std() < pixels.std()


def test_moving_average_on_a_signal_shorter_than_the_window():
    assert moving_average(np.array([1.0, 2.0]), fs=50, seconds=0.1).shape == (2,)


def test_a_pixel_missing_for_the_whole_recording_stays_missing_and_alone():
    pixels = np.random.default_rng(0).normal(size=(100, 2, 2))
    pixels[:, 0, 0] = np.nan
    smoothed = moving_average(pixels, fs=50, seconds=0.1)
    assert np.all(np.isnan(smoothed[:, 0, 0]))
    assert np.all(np.isfinite(smoothed[:, 1, 1]))


def test_nan_at_some_frames_is_rejected():
    x = np.ones(100)
    x[50] = np.nan
    with pytest.raises(ValueError, match="some frames only"):
        moving_average(x, fs=50, seconds=0.1)


# ── respiratory rate ───────────────────────────────────────────────────────────


def test_respiratory_rate_of_a_30_per_minute_sine():
    assert abs(rate(sine(0.5, fs=50, seconds=120), fs=50) * 60 - 30) < 0.5


def test_respiratory_rate_between_spectral_bins_is_refined():
    # 22.2/min sits between two 2/min bins of the 30 s window.
    assert abs(rate(sine(0.37, fs=50, seconds=120), fs=50) * 60 - 22.2) < 0.2


def test_respiratory_rate_ignores_a_smaller_cardiac_component():
    x = sine(0.25, fs=50, seconds=120) + sine(1.5, fs=50, seconds=120, amplitude=0.1)
    assert abs(rate(x, fs=50) * 60 - 15) < 0.5


def test_respiratory_rate_works_on_a_signal_shorter_than_the_window():
    x = sine(0.5, fs=20, seconds=25)  # 25 s < 30 s Welch window
    assert abs(rate(x, fs=20) * 60 - 30) < 2


def test_respiratory_rate_peak_on_the_range_edge_is_returned_unrefined():
    drift = np.linspace(0, 1, 50 * 120)  # slow drift, no breathing
    assert rate(drift, fs=50) * 60 == pytest.approx(CFG.respiratory_rate_min_bpm)


@pytest.mark.parametrize(
    "x, fs, message",
    [
        (np.full(3000, np.nan), 50, "NaN"),
        (np.zeros(3000), 50, "Flat"),
        (sine(0.5, fs=50, seconds=10), 50, "too short"),
        (sine(0.5, fs=50, seconds=60), 0, "positive"),
    ],
)
def test_respiratory_rate_rejects_unusable_input(x, fs, message):
    with pytest.raises(ValueError, match=message):
        rate(x, fs)


def test_respiratory_rate_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="below"):
        respiratory_rate(sine(0.5, 50), 50, min_bpm=60, max_bpm=6, window_seconds=30)


# ── MDN filter ─────────────────────────────────────────────────────────────────


def test_mdn_removes_the_heart_and_keeps_the_breathing():
    fs = 50
    breathing = sine(0.5, fs, seconds=120)
    heart = sine(1.5, fs, seconds=120, amplitude=0.2)
    filtered = mdn(breathing + heart, fs, respiratory_hz=0.5, heart_hz=1.5)
    drop_db = 10 * np.log10(power_near(filtered, fs, 1.5) / power_near(heart, fs, 1.5))
    assert drop_db < -30
    assert np.ptp(filtered[500:-500]) / np.ptp(breathing[500:-500]) > 0.99


def test_mdn_removes_the_second_cardiac_harmonic_at_20_hz():
    fs = 20
    breathing = sine(0.17, fs, seconds=180)
    heart = sine(1.07, fs, seconds=180, amplitude=0.2)
    second = sine(2.14, fs, seconds=180, amplitude=0.1)
    filtered = mdn(breathing + heart + second, fs, respiratory_hz=0.17, heart_hz=1.07)
    drop_db = 10 * np.log10(
        power_near(filtered, fs, 2.14) / power_near(second, fs, 2.14)
    )
    assert drop_db < -30


def test_mdn_filters_pixels_and_1d_signals_along_time():
    pixels = np.random.default_rng(1).normal(size=(1000, 4, 4))
    assert mdn(pixels, 50, 0.5, 1.5).shape == pixels.shape
    assert mdn(pixels[:, 0, 0], 50, 0.5, 1.5).shape == (1000,)


@pytest.mark.parametrize(
    "x, fs, respiratory_hz, heart_hz, message",
    [
        (np.zeros(1000), 50, 0.5, 0.5, "must be above"),
        (np.zeros(1000), 50, 0.5, 0.4, "must be above"),
        (np.zeros(1000), 50, 0.0, 1.5, "positive"),
        (np.zeros(1000), 7, 0.5, 1.5, "too low"),
        (np.zeros(30), 50, 0.5, 1.5, "too short"),
    ],
)
def test_mdn_rejects_unusable_input(x, fs, respiratory_hz, heart_hz, message):
    with pytest.raises(ValueError, match=message):
        mdn(x, fs, respiratory_hz, heart_hz)


def test_mdn_rejects_nan_at_some_frames():
    x = np.ones(1000)
    x[500] = np.nan
    with pytest.raises(ValueError, match="some frames only"):
        mdn(x, 50, 0.5, 1.5)
