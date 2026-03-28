"""Tests for reconstruction.ridge_model — Ridge regression wrapper."""

import numpy as np
import pytest

from fasteit.reconstruction.ridge_model import RidgeReconstructor


@pytest.fixture()
def synthetic_data():
    """Synthetic linear dataset: Y = X @ W_true + noise."""
    rng = np.random.default_rng(42)
    n_frames, n_feat, n_pix = 500, 208, 1024
    W_true = rng.standard_normal((n_feat, n_pix)) * 0.01
    X = rng.standard_normal((n_frames, n_feat))
    Y = X @ W_true + rng.standard_normal((n_frames, n_pix)) * 0.001
    return X, Y, W_true


class TestRidgeReconstructor:

    def test_fit_returns_self(self, synthetic_data):
        X, Y, _ = synthetic_data
        model = RidgeReconstructor(alpha=1.0)
        result = model.fit(X, Y)
        assert result is model

    def test_predict_shape(self, synthetic_data):
        X, Y, _ = synthetic_data
        model = RidgeReconstructor(alpha=1.0).fit(X, Y)
        pred = model.predict(X)
        assert pred.shape == Y.shape

    def test_score_near_one_on_linear_data(self, synthetic_data):
        X, Y, _ = synthetic_data
        model = RidgeReconstructor(alpha=0.01).fit(X, Y)
        r2 = model.score(X, Y)
        assert r2 > 0.95, f"R² = {r2:.4f}, expected > 0.95"

    def test_coef_shape(self, synthetic_data):
        X, Y, _ = synthetic_data
        model = RidgeReconstructor(alpha=1.0).fit(X, Y)
        assert model.coef_.shape == (1024, 208)

    def test_intercept_shape(self, synthetic_data):
        X, Y, _ = synthetic_data
        model = RidgeReconstructor(alpha=1.0).fit(X, Y)
        assert model.intercept_.shape == (1024,)

    def test_n_features_set_after_fit(self, synthetic_data):
        X, Y, _ = synthetic_data
        model = RidgeReconstructor(alpha=1.0).fit(X, Y)
        assert model.n_features == 208
        assert model.n_pixels == 1024

    def test_save_and_load_roundtrip(self, synthetic_data, tmp_path):
        X, Y, _ = synthetic_data
        model = RidgeReconstructor(alpha=2.5).fit(X, Y)
        pred_original = model.predict(X[:10])

        save_path = tmp_path / "model.npz"
        model.save(save_path)

        loaded = RidgeReconstructor.load(save_path)
        pred_loaded = loaded.predict(X[:10])

        np.testing.assert_allclose(pred_loaded, pred_original, atol=1e-10)
        assert loaded.alpha == 2.5
        assert loaded.n_features == 208
        assert loaded.n_pixels == 1024

    def test_higher_alpha_reduces_coef_magnitude(self, synthetic_data):
        X, Y, _ = synthetic_data
        model_low = RidgeReconstructor(alpha=0.01).fit(X, Y)
        model_high = RidgeReconstructor(alpha=100.0).fit(X, Y)
        norm_low = np.linalg.norm(model_low.coef_)
        norm_high = np.linalg.norm(model_high.coef_)
        assert norm_high < norm_low, "Higher alpha should shrink coefficients"

    def test_predict_unfitted_raises(self):
        model = RidgeReconstructor(alpha=1.0)
        X = np.random.default_rng(0).random((10, 208))
        with pytest.raises((AttributeError, RuntimeError)):
            model.predict(X)

    def test_save_unfitted_raises(self, tmp_path):
        model = RidgeReconstructor(alpha=1.0)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.save(tmp_path / "model.npz")

    def test_fit_frame_mismatch_raises(self):
        model = RidgeReconstructor(alpha=1.0)
        X = np.random.default_rng(1).random((100, 208))
        Y = np.random.default_rng(2).random((99, 1024))
        with pytest.raises(ValueError, match="same number of frames"):
            model.fit(X, Y)
