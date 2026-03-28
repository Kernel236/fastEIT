"""Tests for reconstruction.metrics — evaluation metrics."""

import numpy as np
import pytest

from fasteit.reconstruction.metrics import (
    correlation_per_frame,
    error_map,
    global_signal_correlation,
    mse_per_frame,
    r2_score,
    rmse_global,
    summary_metrics,
)


class TestMsePerFrame:

    def test_zero_error(self):
        Y = np.random.default_rng(0).random((10, 1024))
        mse = mse_per_frame(Y, Y)
        np.testing.assert_allclose(mse, 0.0)

    def test_known_error(self):
        Y_true = np.zeros((2, 4))
        Y_pred = np.ones((2, 4))
        mse = mse_per_frame(Y_true, Y_pred)
        np.testing.assert_allclose(mse, [1.0, 1.0])

    def test_shape(self):
        Y = np.random.default_rng(1).random((50, 1024))
        mse = mse_per_frame(Y, Y + 0.01)
        assert mse.shape == (50,)


class TestRmseGlobal:

    def test_zero_error(self):
        Y = np.random.default_rng(10).random((20, 1024))
        assert rmse_global(Y, Y) == pytest.approx(0.0)

    def test_known_value(self):
        Y_true = np.zeros((2, 4))
        Y_pred = np.ones((2, 4))
        # MSE = 1.0, RMSE = sqrt(1.0) = 1.0
        assert rmse_global(Y_true, Y_pred) == pytest.approx(1.0)

    def test_rmse_is_sqrt_of_mean_mse(self):
        rng = np.random.default_rng(11)
        Y_true = rng.random((50, 1024))
        Y_pred = Y_true + rng.normal(0, 0.1, Y_true.shape)
        mse = mse_per_frame(Y_true, Y_pred)
        expected = float(np.sqrt(np.mean(mse)))
        assert rmse_global(Y_true, Y_pred) == pytest.approx(expected, rel=1e-10)


class TestR2Score:

    def test_perfect_prediction(self):
        Y = np.random.default_rng(12).random((50, 1024))
        assert r2_score(Y, Y) == pytest.approx(1.0)

    def test_mean_prediction_gives_zero(self):
        Y_true = np.random.default_rng(13).random((100, 4))
        Y_pred = np.full_like(Y_true, Y_true.mean())
        assert r2_score(Y_true, Y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean_is_negative(self):
        Y_true = np.random.default_rng(14).random((50, 4))
        # Predict the opposite direction from the mean
        Y_pred = 2 * Y_true.mean() - Y_true
        assert r2_score(Y_true, Y_pred) < 0.0

    def test_constant_identical(self):
        Y = np.full((10, 4), 5.0)
        assert r2_score(Y, Y) == pytest.approx(1.0)

    def test_constant_different(self):
        Y_true = np.full((10, 4), 5.0)
        Y_pred = np.full((10, 4), 3.0)
        assert r2_score(Y_true, Y_pred) == pytest.approx(0.0)

    def test_known_r2_value(self):
        Y_true = np.array([[1.0, 2.0, 3.0, 4.0]])
        Y_pred = np.array([[1.0, 2.0, 3.0, 3.0]])  # one pixel off by 1
        # SS_res = 1, SS_tot = 5, R² = 0.8
        assert r2_score(Y_true, Y_pred) == pytest.approx(0.8)


class TestErrorMap:

    def test_shape(self):
        Y = np.random.default_rng(15).random((20, 1024))
        emap = error_map(Y, Y + 0.1)
        assert emap.shape == (32, 32)

    def test_zero_error(self):
        Y = np.random.default_rng(16).random((20, 1024))
        emap = error_map(Y, Y)
        np.testing.assert_allclose(emap, 0.0)

    def test_custom_shape(self):
        Y = np.random.default_rng(17).random((10, 100))
        emap = error_map(Y, Y + 1.0, image_shape=(10, 10))
        assert emap.shape == (10, 10)
        np.testing.assert_allclose(emap, 1.0)


class TestCorrelationPerFrame:

    def test_perfect_correlation(self):
        Y = np.random.default_rng(2).random((10, 1024))
        corr = correlation_per_frame(Y, Y)
        np.testing.assert_allclose(corr, 1.0, atol=1e-10)

    def test_scaled_copy_still_perfect(self):
        Y = np.random.default_rng(3).random((10, 1024))
        corr = correlation_per_frame(Y, Y * 3.0 + 7.0)
        np.testing.assert_allclose(corr, 1.0, atol=1e-10)

    def test_shape(self):
        Y = np.random.default_rng(4).random((20, 1024))
        corr = correlation_per_frame(Y, Y)
        assert corr.shape == (20,)

    def test_constant_frame_returns_one(self):
        Y_true = np.ones((1, 1024))
        Y_pred = np.ones((1, 1024))
        corr = correlation_per_frame(Y_true, Y_pred)
        assert corr.shape == (1,)
        assert corr[0] == pytest.approx(1.0)

    def test_constant_frame_different_constants_returns_zero(self):
        Y_true = np.ones((1, 1024))
        Y_pred = np.full((1, 1024), 2.0)
        corr = correlation_per_frame(Y_true, Y_pred)
        assert corr[0] == pytest.approx(0.0)

    def test_anticorrelated_returns_minus_one(self):
        Y = np.random.default_rng(99).random((5, 1024))
        Y_mean = Y.mean(axis=1, keepdims=True)
        Y_neg = 2 * Y_mean - Y  # reflected around mean
        corr = correlation_per_frame(Y, Y_neg)
        np.testing.assert_allclose(corr, -1.0, atol=1e-10)


class TestGlobalSignalCorrelation:

    def test_identical_signals(self):
        Y = np.random.default_rng(5).random((100, 1024))
        r = global_signal_correlation(Y, Y)
        assert r == pytest.approx(1.0)

    def test_negated_signal(self):
        Y = np.random.default_rng(6).random((100, 1024))
        r = global_signal_correlation(Y, -Y)
        assert r == pytest.approx(-1.0)

    def test_constant_identical_returns_one(self):
        Y = np.ones((100, 1024))
        r = global_signal_correlation(Y, Y)
        assert r == pytest.approx(1.0)

    def test_constant_different_returns_zero(self):
        Y_true = np.ones((100, 1024))
        Y_pred = np.full((100, 1024), 2.0)
        r = global_signal_correlation(Y_true, Y_pred)
        assert r == pytest.approx(0.0)


class TestSummaryMetrics:

    def test_keys_present(self):
        Y = np.random.default_rng(7).random((50, 1024))
        m = summary_metrics(Y, Y)
        expected_keys = {
            "r2", "rmse",
            "mse_mean", "mse_std",
            "corr_spatial_mean", "corr_spatial_std",
            "corr_global",
        }
        assert set(m.keys()) == expected_keys

    def test_perfect_prediction(self):
        Y = np.random.default_rng(8).random((50, 1024))
        m = summary_metrics(Y, Y)
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mse_mean"] == pytest.approx(0.0)
        assert m["corr_spatial_mean"] == pytest.approx(1.0)
        assert m["corr_global"] == pytest.approx(1.0)


class TestInputValidation:

    def test_mismatched_shapes_raise(self):
        Y_true = np.zeros((10, 1024))
        Y_pred = np.zeros((10, 1023))
        with pytest.raises(ValueError, match="identical shape"):
            _ = mse_per_frame(Y_true, Y_pred)

    def test_non_2d_inputs_raise(self):
        Y_true = np.zeros((10, 32, 32))
        Y_pred = np.zeros((10, 32, 32))
        with pytest.raises(ValueError, match="must be 2D arrays"):
            _ = correlation_per_frame(Y_true, Y_pred)

    def test_error_map_shape_mismatch_raises(self):
        Y_true = np.zeros((5, 1024))
        Y_pred = np.zeros((5, 1024))
        with pytest.raises(ValueError, match="image_shape is incompatible"):
            _ = error_map(Y_true, Y_pred, image_shape=(16, 16))
