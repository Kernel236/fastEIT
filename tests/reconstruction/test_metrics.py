"""Tests for reconstruction.metrics - evaluation metrics."""

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

# Two frames with a different error each: 1 and 3 on every pixel.
Y_ZERO = np.zeros((2, 4))
Y_OFF = np.array([[1.0] * 4, [3.0] * 4])


class TestMseAndRmse:
    def test_mse_per_frame(self):
        np.testing.assert_allclose(mse_per_frame(Y_ZERO, Y_OFF), [1.0, 9.0])

    def test_rmse_is_the_root_of_the_mean_over_frames_and_pixels(self):
        assert rmse_global(Y_ZERO, Y_OFF) == pytest.approx(np.sqrt(5.0))


class TestR2Score:
    def test_known_r2_value(self):
        Y_true = np.array([[1.0, 2.0, 3.0, 4.0]])
        Y_pred = np.array([[1.0, 2.0, 3.0, 3.0]])  # one pixel off by 1
        # SS_res = 1, SS_tot = 5, R² = 0.8
        assert r2_score(Y_true, Y_pred) == pytest.approx(0.8)

    def test_mean_prediction_gives_zero(self):
        Y_true = np.random.default_rng(13).random((100, 4))
        Y_pred = np.full_like(Y_true, Y_true.mean())
        assert r2_score(Y_true, Y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean_is_negative(self):
        Y_true = np.random.default_rng(14).random((50, 4))
        Y_pred = 2 * Y_true.mean() - Y_true  # mirrored around the mean
        assert r2_score(Y_true, Y_pred) < 0.0

    def test_constant_target(self):
        Y_true = np.full((10, 4), 5.0)
        assert r2_score(Y_true, Y_true) == 1.0
        assert r2_score(Y_true, np.full((10, 4), 3.0)) == 0.0


class TestErrorMap:
    def test_mae_per_pixel_on_the_32x32_grid(self):
        Y = np.random.default_rng(15).random((20, 1024))
        emap = error_map(Y, Y + 0.1)
        assert emap.shape == (32, 32)
        np.testing.assert_allclose(emap, 0.1)

    def test_custom_shape(self):
        Y = np.random.default_rng(17).random((10, 100))
        emap = error_map(Y, Y + 1.0, image_shape=(10, 10))
        assert emap.shape == (10, 10)
        np.testing.assert_allclose(emap, 1.0)


class TestCorrelationPerFrame:
    def test_scaled_and_shifted_copy_is_perfect(self):
        Y = np.random.default_rng(3).random((10, 1024))
        np.testing.assert_allclose(
            correlation_per_frame(Y, Y * 3.0 + 7.0), 1.0, atol=1e-10
        )

    def test_anticorrelated_returns_minus_one(self):
        Y = np.random.default_rng(99).random((5, 1024))
        Y_neg = 2 * Y.mean(axis=1, keepdims=True) - Y  # mirrored around the mean
        np.testing.assert_allclose(correlation_per_frame(Y, Y_neg), -1.0, atol=1e-10)

    def test_constant_frames(self):
        """Pearson is undefined: 1 if the prediction is the same constant frame, else 0."""
        Y_true = np.ones((3, 1024))
        Y_pred = np.ones((3, 1024))
        Y_pred[1] = 2.0  # another constant
        Y_pred[2] = np.arange(1024)  # not constant
        np.testing.assert_array_equal(
            correlation_per_frame(Y_true, Y_pred), [1.0, 0.0, 0.0]
        )


class TestGlobalSignalCorrelation:
    def test_negated_signal(self):
        Y = np.random.default_rng(6).random((100, 1024))
        assert global_signal_correlation(Y, -Y) == pytest.approx(-1.0)

    def test_constant_signal(self):
        Y = np.ones((100, 1024))
        assert global_signal_correlation(Y, Y) == 1.0
        assert global_signal_correlation(Y, 2.0 * Y) == 0.0


class TestSummaryMetrics:
    def test_perfect_prediction(self):
        Y = np.random.default_rng(8).random((50, 1024))
        m = summary_metrics(Y, Y)
        assert set(m) == {
            "r2",
            "rmse",
            "mse_mean",
            "mse_std",
            "corr_spatial_mean",
            "corr_spatial_std",
            "corr_global",
        }
        assert m["r2"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mse_mean"] == pytest.approx(0.0)
        assert m["corr_spatial_mean"] == pytest.approx(1.0)
        assert m["corr_global"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "shapes",
    [((10, 1024), (10, 1023)), ((10, 32, 32), (10, 32, 32))],
    ids=["different shapes", "not 2D"],
)
def test_wrong_shapes_raise(shapes):
    Y_true, Y_pred = (np.zeros(s) for s in shapes)
    with pytest.raises(ValueError, match="2D arrays"):
        mse_per_frame(Y_true, Y_pred)
