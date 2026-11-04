"""The reconstruction model: channel statistics + standardised ridge, fitted on whole recordings.

``fit_model`` takes raw paired recordings and does every step of ``data_prep``: reference at
the bottom of each recording, relative change of X with the altered channels set to 0 (typical
reference and usual sign of every channel from these recordings), Y minus its reference at the
same frames, then the ridge. ``EITReconstructionModel.predict`` repeats the same normalisation
on a new recording (X only) and returns its images, without the low-pass.

The ridge (``StandardisedRidge``) maps ``x (N, 208) -> pixels (N, 1024)``. X and Y are
standardised on the training data (mean 0, standard deviation 1 per channel and per pixel),
each standardised pixel is a linear combination of the standardised channels with an L2
penalty, and the prediction is brought back to pixel units. It is solved in closed form,
``W = (X^T X + alpha I)^-1 X^T Y``, with ``X^T X`` and ``X^T Y`` accumulated over chunks of
frames in float64: no full copy of Y (the largest array) is made.

Device-specific: Dräger PulmoVista 500, 16 electrodes, adjacent drive. The saved file holds a
trained model: it stays local (``*.npz`` is ignored by git).

Reference for Ridge regression:
    Hoerl AE, Kennard RW. "Ridge Regression: Biased Estimation for
    Nonorthogonal Problems." *Technometrics* 12(1), 1970, pp. 55-67.
    DOI: 10.1080/00401706.1970.10488634
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from fasteit.reconstruction.data_prep import (
    channel_stats,
    normalize,
    normalize_relative,
    reference,
    reference_frames,
)

_CHUNK = 50_000  # frames per chunk


class StandardisedRidge:
    """Standardisation of X and Y + ridge regression.

    Attributes:
        alpha: Regularisation strength (higher = more regularisation).
        n_features: Number of input features (set after fit).
        n_pixels: Number of output pixels (set after fit).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        self._coef: np.ndarray | None = None
        self.n_features: int | None = None
        self.n_pixels: int | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> StandardisedRidge:
        """Fit the scalers and the ridge on normalised training data.

        Args:
            X: Normalised input, shape ``(N_frames, n_features)``.
            Y: Normalised pixel targets, shape ``(N_frames, n_pixels)``.

        Returns:
            self (for chaining).
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of frames. "
                f"Got X={X.shape[0]}, Y={Y.shape[0]}."
            )
        chunks = [slice(i, i + _CHUNK) for i in range(0, X.shape[0], _CHUNK)]
        # New scalers: a refit starts from zero.
        self._scaler_x, self._scaler_y = StandardScaler(), StandardScaler()
        for c in chunks:
            self._scaler_x.partial_fit(X[c])
            self._scaler_y.partial_fit(Y[c])
        xtx = np.zeros((X.shape[1], X.shape[1]))
        xty = np.zeros((X.shape[1], Y.shape[1]))
        for c in chunks:
            xs = self._scaler_x.transform(X[c]).astype(np.float64)
            xtx += xs.T @ xs
            xty += xs.T @ self._scaler_y.transform(Y[c]).astype(np.float64)
        xtx[np.diag_indices_from(xtx)] += self.alpha
        self._coef = np.linalg.solve(xtx, xty).T
        self.n_pixels, self.n_features = self._coef.shape
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict pixel values from normalised input.

        Args:
            X: Normalised input, shape ``(N_frames, n_features)``.

        Returns:
            Predicted pixels, shape ``(N_frames, n_pixels)``.
        """
        if self._coef is None:
            raise RuntimeError(
                "Model must be fitted before predicting. Call fit() first."
            )
        return self._scaler_y.inverse_transform(
            self._scaler_x.transform(X) @ self._coef.T
        )

    @property
    def coef_(self) -> np.ndarray:
        """Weight matrix W in standardised units, shape ``(n_pixels, n_features)``."""
        return self._coef

    def state(self) -> dict[str, np.ndarray]:
        """Weights, scalers and alpha as arrays, for ``np.savez`` (no pickle)."""
        if self._coef is None:
            raise RuntimeError("Model must be fitted before saving. Call fit() first.")
        return {
            "coef": self._coef,
            "x_mean": self._scaler_x.mean_,
            "x_scale": self._scaler_x.scale_,
            "y_mean": self._scaler_y.mean_,
            "y_scale": self._scaler_y.scale_,
            "meta": np.array(json.dumps({"alpha": self.alpha})),
        }

    @classmethod
    def from_state(cls, data) -> StandardisedRidge:
        """A fitted ridge from the arrays of ``state()`` (a dict or a loaded ``.npz``)."""
        ridge = cls(alpha=json.loads(str(data["meta"]))["alpha"])
        ridge._coef = data["coef"]
        for scaler, prefix in ((ridge._scaler_x, "x"), (ridge._scaler_y, "y")):
            scaler.mean_ = data[f"{prefix}_mean"]
            scaler.scale_ = data[f"{prefix}_scale"]
            scaler.var_ = scaler.scale_**2
            scaler.n_features_in_ = scaler.mean_.shape[0]
        ridge.n_pixels, ridge.n_features = ridge._coef.shape
        return ridge


@dataclass
class EITReconstructionModel:
    """Ridge plus the statistics of the channels it was trained with.

    Attributes:
        ridge: Fitted ridge (standardisation inside).
        typical_abs: Typical |reference| per channel, from the training recordings.
        usual_sign: Usual sign of the reference per channel, from the training recordings.
    """

    ridge: StandardisedRidge
    typical_abs: np.ndarray
    usual_sign: np.ndarray

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        """Relative change of one whole recording, altered channels set to 0.

        Args:
            x: Raw transimpedances of one recording, shape ``(N_frames, channels)``.

        Returns:
            Normalised recording, same shape.
        """
        return normalize_relative(
            x, reference_frames(x), self.typical_abs, self.usual_sign
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Images of one whole recording, zero at its bottom, without the low-pass.

        Args:
            x: Raw transimpedances of one recording, shape ``(N_frames, channels)``.

        Returns:
            Pixels, shape ``(N_frames, n_pixels)``.
        """
        return self.ridge.predict(self.normalize_x(x))

    def save(self, path: str | Path) -> None:
        """Save ridge and channel statistics to one ``.npz`` file (no pickle)."""
        np.savez(
            Path(path),
            typical_abs=self.typical_abs,
            usual_sign=self.usual_sign,
            **self.ridge.state(),
        )

    @classmethod
    def load(cls, path: str | Path) -> EITReconstructionModel:
        """Load a model saved by ``save``."""
        data = np.load(path, allow_pickle=False)
        return cls(
            StandardisedRidge.from_state(data), data["typical_abs"], data["usual_sign"]
        )


def fit_model(
    recordings: Sequence[tuple[np.ndarray, np.ndarray]], alpha: float = 0.1
) -> EITReconstructionModel:
    """Fit the whole model on raw paired recordings.

    Args:
        recordings: One ``(x, y)`` per recording: raw transimpedances ``(N, channels)`` and
            raw pixels ``(N, n_pixels)``, already chosen as training data.
        alpha: Ridge penalty.

    Returns:
        The fitted model.
    """
    frames = [reference_frames(x) for x, _ in recordings]
    refs = np.stack(
        [reference(x, f) for (x, _), f in zip(recordings, frames, strict=True)]
    )
    typical_abs, usual_sign = channel_stats(refs)
    # Filled recording by recording: no list of normalised copies next to the result.
    n_total = sum(len(x) for x, _ in recordings)
    X = np.empty((n_total, recordings[0][0].shape[1]), dtype=np.float32)
    Y = np.empty((n_total, recordings[0][1].shape[1]), dtype=np.float32)
    start = 0
    for (x, y), f in zip(recordings, frames, strict=True):
        stop = start + len(x)
        X[start:stop] = normalize_relative(x, f, typical_abs, usual_sign)
        Y[start:stop] = normalize(y, f)[0]
        start = stop
    return EITReconstructionModel(
        StandardisedRidge(alpha).fit(X, Y), typical_abs, usual_sign
    )
