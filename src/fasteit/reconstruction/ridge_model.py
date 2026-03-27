"""Ridge regression model for EIT image reconstruction.

Learns the mapping ``Δvv (N, 208) → Δpixels (N, 1024)`` from paired
Dräger .eit/.bin recordings.  Each pixel is a linear combination of the
208 transimpedance measurements, regularised with an L2 penalty (Ridge).

Device-specific: Dräger PulmoVista 500, 16 electrodes, adjacent drive.

Reference for Ridge regression:
    Hoerl AE, Kennard RW. "Ridge Regression: Biased Estimation for
    Nonorthogonal Problems." *Technometrics* 12(1), 1970, pp. 55–67.
    DOI: 10.1080/00401706.1970.10488634
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import Ridge

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


def _check_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for Ridge reconstruction. "
            "Install it with: pip install scikit-learn"
        )


class RidgeReconstructor:
    """Ridge regression wrapper for EIT reconstruction.

    Attributes:
        alpha: Regularisation strength (higher = more regularisation).
        n_features: Number of input features (set after fit).
        n_pixels: Number of output pixels (set after fit).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        _check_sklearn()
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self.n_features: int | None = None
        self.n_pixels: int | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> RidgeReconstructor:
        """Fit the model on normalised training data.

        Args:
            X: Normalised input, shape ``(N_frames, n_features)``.
            Y: Normalised pixel targets, shape ``(N_frames, 1024)``.

        Returns:
            self (for chaining).
        """
        self._model.fit(X, Y)
        self.n_features = X.shape[1]
        self.n_pixels = Y.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict pixel values from normalised input.

        Args:
            X: Normalised input, shape ``(N_frames, n_features)``.

        Returns:
            Predicted pixels, shape ``(N_frames, n_pixels)``.
        """
        return self._model.predict(X)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Return R² score on the given data.

        R² = 1 means perfect prediction; R² = 0 means the model
        predicts no better than the mean.

        Args:
            X: Normalised input, shape ``(N_frames, n_features)``.
            Y: Normalised pixel targets, shape ``(N_frames, n_pixels)``.

        Returns:
            R² coefficient of determination.
        """
        return float(self._model.score(X, Y))

    @property
    def coef_(self) -> np.ndarray:
        """Weight matrix W, shape ``(n_pixels, n_features)``."""
        return self._model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Bias vector b, shape ``(n_pixels,)``."""
        return self._model.intercept_

    def save(self, path: str | Path) -> None:
        """Save model weights to a ``.npz`` file.

        Saves ``coef_``, ``intercept_``, and ``alpha`` so the model
        can be reconstructed without pickle (portable, auditable).

        Args:
            path: Output file path (recommended: ``.npz`` extension).
        """
        path = Path(path)
        meta = {"alpha": self.alpha}
        np.savez(
            path,
            coef=self._model.coef_,
            intercept=self._model.intercept_,
            meta=json.dumps(meta),
        )

    @classmethod
    def load(cls, path: str | Path) -> RidgeReconstructor:
        """Load model weights from a ``.npz`` file.

        Args:
            path: Path to saved ``.npz`` file.

        Returns:
            A fitted ``RidgeReconstructor`` ready for ``predict()``.
        """
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        model = cls(alpha=meta["alpha"])
        model._model.coef_ = data["coef"]
        model._model.intercept_ = data["intercept"]
        model.n_features = data["coef"].shape[1]
        model.n_pixels = data["coef"].shape[0]
        # sklearn needs intercept_ set + is_fitted flag
        model._model.n_features_in_ = model.n_features
        return model
