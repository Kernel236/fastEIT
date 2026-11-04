"""Tests for reconstruction.model - standardised ridge and the whole model (channel statistics + ridge)."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from fasteit.reconstruction import model as model_module
from fasteit.reconstruction.metrics import r2_score
from fasteit.reconstruction.model import (
    EITReconstructionModel,
    StandardisedRidge,
    fit_model,
)

N_CHANNELS, N_PIXELS = 208, 1024
W_TRUE = np.random.default_rng(0).standard_normal((N_CHANNELS, N_PIXELS))
SIGNS = np.random.default_rng(2).choice([-1.0, 1.0], N_CHANNELS)
BASE = np.random.default_rng(1).uniform(1.0, 10.0, N_CHANNELS) * SIGNS


def recording(seed, n_frames=600):
    """Raw (x, y) of one recording: y is a linear map of the relative change of x + offset."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / 50
    volume = (
        0.5
        - 0.5 * np.cos(2 * np.pi * t / rng.uniform(3, 5))
        + rng.uniform(0, 0.5) * t / t[-1]
    )
    gain = rng.uniform(0.5, 2.0, N_CHANNELS)  # contact and belt: a factor per channel
    # each channel moves a little differently
    pattern = 1 + 0.03 * rng.standard_normal(N_CHANNELS)
    x = BASE * gain * (1 + 0.05 * volume[:, None] * pattern)
    relative = 0.05 * (volume - volume.min())[:, None] * pattern
    y = relative @ W_TRUE + rng.uniform(-50, 50)
    return x, y


def r2_changes(y, y_pred):
    return r2_score(y - y.mean(axis=0), y_pred - y_pred.mean(axis=0))


class TestEITReconstructionModel:
    def test_fit_and_predict_recover_a_linear_map(self):
        model = fit_model([recording(seed) for seed in range(10, 16)])
        x, y = recording(99)
        assert r2_changes(y, model.predict(x)) > 0.99

    def test_save_load_roundtrip(self, tmp_path):
        model = fit_model([recording(seed) for seed in range(10, 13)])
        model.save(tmp_path / "model.npz")
        loaded = EITReconstructionModel.load(tmp_path / "model.npz")
        x, _ = recording(98)
        np.testing.assert_allclose(loaded.predict(x), model.predict(x))
        np.testing.assert_array_equal(loaded.typical_abs, model.typical_abs)
        np.testing.assert_array_equal(loaded.usual_sign, model.usual_sign)

    def test_bad_channel_of_a_new_recording_is_zero(self):
        model = fit_model([recording(seed) for seed in range(10, 13)])
        x, _ = recording(97)
        x[:, 3] = 1e-6 * np.sign(BASE[3])  # reference close to 0: a detached electrode
        out = model.normalize_x(x)
        assert np.all(out[:, 3] == 0.0)
        assert np.any(out[:, 4] != 0.0)


@pytest.fixture()
def synthetic_data():
    """Synthetic linear dataset: Y = X @ W_true + noise."""
    rng = np.random.default_rng(42)
    n_frames, n_feat, n_pix = 500, 208, 1024
    W_true = rng.standard_normal((n_feat, n_pix)) * 0.01
    X = rng.standard_normal((n_frames, n_feat))
    Y = X @ W_true + rng.standard_normal((n_frames, n_pix)) * 0.001
    return X, Y


class TestStandardisedRidge:
    def test_standardises_x_and_y_inside(self, monkeypatch):
        """Scale X and Y on the training data, ridge, back to pixels.

        Small chunks, so the sums X^T X and X^T Y are accumulated over several of them.
        """
        monkeypatch.setattr(model_module, "_CHUNK", 64)
        rng = np.random.default_rng(7)
        # channels on very different scales
        X = rng.standard_normal((300, 20)) * rng.uniform(1e-3, 1e3, 20)
        Y = X @ rng.standard_normal((20, 30)) + 50.0
        sx, sy = StandardScaler().fit(X), StandardScaler().fit(Y)
        ridge = Ridge(alpha=5.0).fit(sx.transform(X), sy.transform(Y))
        expected = sy.inverse_transform(ridge.predict(sx.transform(X)))
        model = StandardisedRidge(alpha=5.0).fit(X, Y)
        np.testing.assert_allclose(model.predict(X), expected, rtol=1e-6, atol=1e-6)

    def test_refit_forgets_the_previous_data(self):
        rng = np.random.default_rng(8)
        X1, X2 = rng.standard_normal((200, 10)), rng.standard_normal((200, 10)) * 5 + 3
        Y1, Y2 = rng.standard_normal((200, 4)), X2 @ rng.standard_normal((10, 4)) + 7
        refitted = StandardisedRidge(alpha=1.0).fit(X1, Y1).fit(X2, Y2)
        fresh = StandardisedRidge(alpha=1.0).fit(X2, Y2)
        np.testing.assert_allclose(refitted.predict(X2), fresh.predict(X2), rtol=1e-10)

    def test_shapes_after_fit(self, synthetic_data):
        model = StandardisedRidge(alpha=1.0).fit(*synthetic_data)
        assert model.coef_.shape == (1024, 208)
        assert model.n_features == 208
        assert model.n_pixels == 1024

    def test_higher_alpha_reduces_coef_magnitude(self, synthetic_data):
        model_low = StandardisedRidge(alpha=0.01).fit(*synthetic_data)
        model_high = StandardisedRidge(alpha=100.0).fit(*synthetic_data)
        assert np.linalg.norm(model_high.coef_) < np.linalg.norm(model_low.coef_)

    def test_unfitted_model_raises(self):
        model = StandardisedRidge(alpha=1.0)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(np.zeros((10, 208)))
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.state()

    def test_fit_frame_mismatch_raises(self):
        model = StandardisedRidge(alpha=1.0)
        with pytest.raises(ValueError, match="same number of frames"):
            model.fit(np.zeros((100, 208)), np.zeros((99, 1024)))
