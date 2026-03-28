"""Tests for reconstruction.data_prep — normalisation and paired loading."""

import numpy as np
import pytest

from fasteit.reconstruction.data_prep import normalize

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
