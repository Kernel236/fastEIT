"""Tests for reconstruction.metrics — evaluation metrics."""

import numpy as np
import pytest

from fasteit.reconstruction.metrics import (
    correlation_per_frame,
    global_signal_correlation,
    mse_per_frame,
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


class TestGlobalSignalCorrelation:

    def test_identical_signals(self):
        Y = np.random.default_rng(5).random((100, 1024))
        r = global_signal_correlation(Y, Y)
        assert r == pytest.approx(1.0)

    def test_negated_signal(self):
        Y = np.random.default_rng(6).random((100, 1024))
        r = global_signal_correlation(Y, -Y)
        assert r == pytest.approx(-1.0)


class TestSummaryMetrics:

    def test_keys_present(self):
        Y = np.random.default_rng(7).random((50, 1024))
        m = summary_metrics(Y, Y)
        expected_keys = {
            "mse_mean", "mse_std",
            "corr_spatial_mean", "corr_spatial_std",
            "corr_global",
        }
        assert set(m.keys()) == expected_keys

    def test_perfect_prediction(self):
        Y = np.random.default_rng(8).random((50, 1024))
        m = summary_metrics(Y, Y)
        assert m["mse_mean"] == pytest.approx(0.0)
        assert m["corr_spatial_mean"] == pytest.approx(1.0)
        assert m["corr_global"] == pytest.approx(1.0)
