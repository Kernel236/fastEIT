"""Tests for the tidal volume of each breath, on synthetic volume-control curves."""

import numpy as np
import pytest

from fasteit.preprocessing.tidal_volume import (
    inspiration_windows,
    volumes_from_flow_curve,
    volumes_from_volume_curve,
)


def vcv_flow(volumes, fs=50, period=4.0, inspiration=1.0, pause=0.5, tau=0.3):
    """Volume-control flow (L/s): constant inspiratory flow, pause, passive expiration.

    Each breath delivers its volume (L) and breathes it out with time constant ``tau``.
    Returns the flow and the frames of start, end of inspiration and end of each breath.
    """
    t = np.arange(0, period, 1 / fs)
    breaths = []
    for volume in volumes:
        flow = np.zeros(len(t))
        flow[t < inspiration] = volume / inspiration
        out = t >= inspiration + pause
        expiration = np.exp(-(t[out] - inspiration - pause) / tau)
        flow[out] = -volume * fs * expiration / expiration.sum()  # all of it, sampled
        breaths.append(flow)
    n = len(t)
    starts = np.arange(len(volumes)) * n
    return np.concatenate(breaths), starts, starts + int(inspiration * fs), starts + n


def volume_curve(flow, fs, starts=None):
    """Volume (L) at each frame, from 0; set back to zero at each start if given."""
    parts = [flow] if starts is None else np.split(flow, starts[1:])
    return np.concatenate([np.r_[0, np.cumsum(part)[:-1]] / fs for part in parts])


def windows(starts, peaks, ends, n, lag=3):
    """Windows of EIT breaths whose valley falls ``lag`` frames after the flow starts."""
    return inspiration_windows(starts + lag, peaks + lag, ends + lag, n)


def test_the_window_holds_the_whole_inspiration_and_nothing_of_the_next():
    first, after_last = inspiration_windows(
        np.array([100, 300]), np.array([150, 350]), np.array([300, 500]), 1000
    )
    # Half an expiration (75 frames) before the start, to the middle of the expiration.
    assert first.tolist() == [25, 225] and after_last.tolist() == [226, 426]


def test_the_window_stays_inside_the_recording():
    first, after_last = inspiration_windows(
        np.array([10]), np.array([60]), np.array([200]), 100
    )
    assert first.tolist() == [0] and after_last.tolist() == [100]


@pytest.mark.parametrize("reset", [False, True])
def test_volume_curve_gives_the_inspiratory_volume(reset):
    flow, starts, peaks, ends = vcv_flow([0.45, 0.50, 0.40])
    volume = volume_curve(flow, 50, starts if reset else None) * 1000  # mL
    first, after_last = windows(starts, peaks, ends, len(flow))
    volumes = volumes_from_volume_curve(volume, first, after_last)
    np.testing.assert_allclose(volumes, [450, 500, 400], atol=1)


def test_flow_curve_gives_the_inspiratory_volume():
    flow, starts, peaks, ends = vcv_flow([0.45, 0.50, 0.40])
    first, after_last = windows(starts, peaks, ends, len(flow))
    volumes = volumes_from_flow_curve(flow, 50, first, after_last)
    np.testing.assert_allclose(volumes, [450, 500, 400], atol=1)


@pytest.mark.parametrize("fs", [20, 50])
def test_both_sampling_rates(fs):
    flow, starts, peaks, ends = vcv_flow([0.45] * 3, fs=fs)
    first, after_last = windows(starts, peaks, ends, len(flow), lag=1)
    np.testing.assert_allclose(
        volumes_from_flow_curve(flow, fs, first, after_last), 450, atol=1
    )


def test_a_leak_lowers_the_expiration_not_the_inspiratory_volume():
    flow, starts, peaks, ends = vcv_flow([0.45] * 3)
    flow[flow < 0] *= 0.8  # 20 % of the volume leaks out of the circuit
    first, after_last = windows(starts, peaks, ends, len(flow))
    np.testing.assert_allclose(
        volumes_from_flow_curve(flow, 50, first, after_last), 450, atol=1
    )
    volume = volume_curve(flow, 50) * 1000  # drifts up, breath after breath
    np.testing.assert_allclose(
        volumes_from_volume_curve(volume, first, after_last), 450, atol=1
    )


def test_missing_values_give_nan_for_that_breath_only():
    flow, starts, peaks, ends = vcv_flow([0.45] * 3)
    flow[starts[1] + 10] = np.nan
    first, after_last = windows(starts, peaks, ends, len(flow))
    volumes = volumes_from_flow_curve(flow, 50, first, after_last)
    assert np.isnan(volumes[1]) and np.isfinite(volumes[[0, 2]]).all()
