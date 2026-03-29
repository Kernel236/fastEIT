"""Tests for reconstruction.data_prep — normalisation and paired loading."""

import numpy as np
import pytest

from fasteit.reconstruction.data_prep import normalize, normalize_rolling

# ── normalize ────────────────────────────────────────────────────────────


class TestNormalize:
    """Unit tests for baseline-subtraction normalisation."""

    def test_output_shape_matches_input(self):
        arr = np.random.default_rng(0).random((100, 208))
        delta, ref = normalize(arr, n_ref=10)
        assert delta.shape == arr.shape
        assert ref.shape == (208,)

    def test_first_n_ref_frames_average_to_zero(self):
        arr = np.random.default_rng(1).random((100, 208))
        delta, _ = normalize(arr, n_ref=10)
        # Mean of first 10 frames should be ~zero (within float precision)
        assert np.allclose(delta[:10].mean(axis=0), 0.0, atol=1e-12)

    def test_ref_equals_mean_of_first_n_ref(self):
        arr = np.random.default_rng(2).random((100, 208))
        n_ref = 20
        delta, ref = normalize(arr, n_ref=n_ref)
        expected_ref = arr[:n_ref].mean(axis=0)
        np.testing.assert_array_equal(ref, expected_ref)

    def test_delta_preserves_differences(self):
        """Differences between consecutive frames are unchanged."""
        arr = np.random.default_rng(3).random((100, 208))
        delta, _ = normalize(arr, n_ref=10)
        np.testing.assert_allclose(
            np.diff(delta, axis=0),
            np.diff(arr, axis=0),
        )

    def test_constant_input_gives_zero_delta(self):
        arr = np.full((50, 4), 42.0)
        delta, ref = normalize(arr, n_ref=10)
        assert np.all(delta == 0.0)
        assert np.all(ref == 42.0)

    def test_n_ref_default_is_50(self):
        arr = np.ones((100, 4))
        arr[50:] = 2.0  # second half is different
        delta, ref = normalize(arr)  # default n_ref=50
        # ref should be mean of first 50 frames (all 1.0)
        np.testing.assert_array_equal(ref, 1.0)

    def test_single_feature_column(self):
        arr = np.arange(20, dtype=float).reshape(20, 1)
        delta, ref = normalize(arr, n_ref=5)
        assert ref.shape == (1,)
        assert ref[0] == pytest.approx(2.0)  # mean of [0,1,2,3,4]
        assert delta[0, 0] == pytest.approx(-2.0)
        assert delta[5, 0] == pytest.approx(3.0)

    def test_n_ref_exceeds_frames_raises(self):
        arr = np.random.default_rng(10).random((20, 4))
        with pytest.raises(ValueError, match="n_ref=50 exceeds"):
            normalize(arr, n_ref=50)


# ── normalize_rolling ──────────────────────────────────────────────────


class TestNormalizeRolling:
    """Unit tests for rolling baseline normalisation."""

    def test_output_shape_matches_input(self):
        arr = np.random.default_rng(20).random((200, 208))
        delta = normalize_rolling(arr, window=50)
        assert delta.shape == arr.shape

    def test_constant_input_gives_zero(self):
        arr = np.full((100, 4), 42.0)
        delta = normalize_rolling(arr, window=20)
        np.testing.assert_allclose(delta, 0.0, atol=1e-12)

    def test_linear_drift_removed(self):
        """A linear ramp (slow drift) should be mostly removed."""
        n = 500
        ramp = np.linspace(0, 10, n).reshape(n, 1)
        delta = normalize_rolling(ramp, window=100)
        # After removing the rolling mean, the residual should be
        # much smaller than the original ramp amplitude (10.0).
        # Interior frames (away from edges) should be near zero.
        interior = delta[100:-100]
        assert np.abs(interior).max() < 0.5

    def test_fast_oscillation_preserved(self):
        """A fast sine wave (breathing) should survive the rolling subtraction."""
        n = 1000
        t = np.arange(n, dtype=float)
        # ~25 breaths/min at 50 Hz = period ~120 frames
        breathing = np.sin(2 * np.pi * t / 120).reshape(n, 1)
        delta = normalize_rolling(breathing, window=250)
        # The breathing amplitude should be mostly preserved (>80%)
        assert delta.std() > 0.8 * breathing.std()

    def test_window_exceeds_frames_raises(self):
        arr = np.random.default_rng(21).random((20, 4))
        with pytest.raises(ValueError, match="window=50 exceeds"):
            normalize_rolling(arr, window=50)

    def test_window_zero_raises(self):
        arr = np.random.default_rng(22).random((20, 4))
        with pytest.raises(ValueError, match="window must be >= 1"):
            normalize_rolling(arr, window=0)
