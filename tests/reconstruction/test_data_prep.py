"""Tests for reconstruction.data_prep - reference frames, subtraction, relative normalisation."""

import numpy as np
import pytest

from fasteit.reconstruction.data_prep import (
    MIN_REFERENCE_FRAMES,
    TRAINING_RMS_MAX,
    VALLEY_FRACTION,
    bad_channels,
    channel_stats,
    load_paired,
    normalize,
    normalize_relative,
    pixel_rms,
    reference_frames,
)


def breathing(n_frames=300, n_channels=8, seed=0):
    """Channels with different scales and signs that all grow in magnitude with inspiration."""
    rng = np.random.default_rng(seed)
    # 0 at end-expiration, 1 at end-inspiration
    volume = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_frames) / 50)
    base = rng.uniform(1.0, 10.0, n_channels) * rng.choice([-1.0, 1.0], n_channels)
    return base * (1 + 0.05 * volume[:, None]), volume


class TestReferenceFrames:
    def test_bottom_of_the_whole_recording(self):
        """Dräger's zero is the lowest end-expiratory level of the recording, wherever it is."""
        x, volume = breathing(n_frames=1000)
        x[600:] *= 0.9  # the lung empties after frame 600: every |V| falls
        frames = reference_frames(x)
        assert frames.min() >= 600
        assert volume[frames].max() < 0.1  # end-expirations

    def test_fraction_and_minimum_number_of_frames(self):
        x, _ = breathing(n_frames=1000)
        assert len(reference_frames(x)) == int(VALLEY_FRACTION * 1000)
        short, _ = breathing(n_frames=89)  # sampled recordings can be this short
        assert len(reference_frames(short)) == MIN_REFERENCE_FRAMES

    def test_a_few_exploding_channels_do_not_move_the_valleys(self):
        x, volume = breathing(n_channels=20)
        x[:, :2] = np.random.default_rng(1).normal(0, 1e-6, (len(x), 2))  # reference ~0
        assert volume[reference_frames(x)].max() < 0.1


class TestNormalize:
    """Subtraction of the reference, used for Y."""

    def test_subtracts_the_reference(self):
        arr = np.arange(20, dtype=float).reshape(20, 1)
        delta, ref = normalize(arr, np.array([0, 2, 4]))
        assert ref[0] == pytest.approx(2.0)
        assert delta[0, 0] == pytest.approx(-2.0)
        assert delta[5, 0] == pytest.approx(3.0)

    def test_delta_preserves_differences(self):
        """Differences between frames (TIV, dEELI) are unchanged."""
        arr = np.random.default_rng(3).random((100, 208))
        delta, _ = normalize(arr, np.arange(10))
        np.testing.assert_allclose(np.diff(delta, axis=0), np.diff(arr, axis=0))


class TestChannelStats:
    def test_typical_is_median_abs_and_sign_is_majority(self):
        # 3 recordings, 2 channels
        refs = np.array([[1.0, -2.0], [3.0, -4.0], [-5.0, 6.0]])
        typical, usual_sign = channel_stats(refs)
        np.testing.assert_array_equal(typical, [3.0, 4.0])
        np.testing.assert_array_equal(usual_sign, [1.0, -1.0])

    def test_tie_counts_as_positive(self):
        _, usual_sign = channel_stats(np.array([[1.0], [-1.0]]))
        np.testing.assert_array_equal(usual_sign, [1.0])


class TestBadChannels:
    def test_small_or_flipped_reference_is_bad(self):
        typical = np.array([1.0, 1.0, 1.0, 1.0])
        usual_sign = np.array([1.0, 1.0, 1.0, -1.0])
        # ok, too small, flipped, ok (negative is the usual sign there)
        ref = np.array([1.0, 0.1, -1.0, -1.0])
        np.testing.assert_array_equal(
            bad_channels(ref, typical, usual_sign), [False, True, True, False]
        )

    def test_threshold_is_a_fraction_of_the_typical_reference(self):
        typical, usual_sign = np.array([1.0, 1.0]), np.array([1.0, 1.0])
        ref = np.array([0.21, 0.19])
        np.testing.assert_array_equal(
            bad_channels(ref, typical, usual_sign, frac=0.2), [False, True]
        )


class TestNormalizeRelative:
    """Relative change, used for X."""

    def test_good_channels_are_change_over_reference(self):
        x = np.array([[2.0, 4.0], [2.0, 4.0], [3.0, 2.0]])
        out = normalize_relative(x, np.array([0, 1]), np.array([2.0, 4.0]), np.ones(2))
        np.testing.assert_allclose(out, [[0.0, 0.0], [0.0, 0.0], [0.5, -0.5]])

    def test_bad_channels_are_zero_in_every_frame(self):
        rng = np.random.default_rng(4)
        x = rng.random((50, 3)) + 1.0
        x[:, 1] *= 1e-3  # reference far below the typical one
        out = normalize_relative(x, np.arange(10), np.ones(3), np.ones(3))
        assert np.all(out[:, 1] == 0.0)
        assert np.all(out[:, [0, 2]] != 0.0)

    def test_scale_of_a_channel_cancels(self):
        """Contact, thorax size and gain multiply a channel: the relative change does not move."""
        x = np.random.default_rng(5).random((40, 2)) + 1.0
        frames = np.arange(10)
        out = normalize_relative(x, frames, np.ones(2), np.ones(2))
        scaled = normalize_relative(
            x * [1.0, 7.0], frames, np.array([1.0, 7.0]), np.ones(2)
        )
        np.testing.assert_allclose(out, scaled)


class TestLoadPaired:
    @staticmethod
    def fake_files(monkeypatch, n_eit, n_bin):
        from types import SimpleNamespace

        from fasteit.reconstruction import data_prep

        eit = SimpleNamespace(measurements=np.ones((n_eit, 208)))
        bin_ = SimpleNamespace(pixels=np.ones((n_bin, 32, 32)), n_frames=n_bin)
        monkeypatch.setattr(
            data_prep,
            "load_data",
            lambda path: eit if str(path).endswith(".eit") else bin_,
        )

    def test_one_frame_difference_is_trimmed(self, monkeypatch):
        self.fake_files(monkeypatch, 101, 100)
        x, y = load_paired("a.eit", "a_01.bin")
        assert x.shape == (100, 208)
        assert y.shape == (100, 1024)

    def test_larger_difference_raises(self, monkeypatch):
        self.fake_files(monkeypatch, 101, 99)
        with pytest.raises(ValueError, match="Frame count mismatch"):
            load_paired("a.eit", "a_01.bin")


class TestPixelRms:
    def test_rms_of_all_pixels_and_frames(self):
        assert pixel_rms(np.full((10, 4), 3.0)) == pytest.approx(3.0)

    def test_local_anomaly_that_cancels_in_the_global_signal(self):
        """Huge values of opposite sign in two pixels: global signal normal, pixel energy high."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, (200, 1024))
        anomalous = y.copy()
        anomalous[:, 0] += 1e3
        anomalous[:, 1] -= 1e3
        assert np.allclose(anomalous.sum(axis=1), y.sum(axis=1))
        assert pixel_rms(anomalous) > TRAINING_RMS_MAX * pixel_rms(y)
