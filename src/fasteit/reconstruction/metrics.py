"""Evaluation metrics for EIT image reconstruction quality.

Compares predicted pixel arrays against ground-truth (.bin) to quantify
how well the Ridge model reproduces the Dräger reconstruction.

All functions take arrays only, so they work on saved predictions too.
"""

from __future__ import annotations

import numpy as np


def _check_shapes(Y_true: np.ndarray, Y_pred: np.ndarray) -> None:
    """Both arrays must be ``(N_frames, n_pixels)``, of the same shape."""
    if Y_true.ndim != 2 or Y_true.shape != Y_pred.shape:
        raise ValueError(
            "Y_true and Y_pred must be 2D arrays (N_frames, n_pixels) of identical "
            f"shape. Got {Y_true.shape} and {Y_pred.shape}."
        )


def mse_per_frame(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Mean squared error per frame.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        MSE per frame, shape ``(N_frames,)``.
    """
    _check_shapes(Y_true, Y_pred)
    return np.mean((Y_true - Y_pred) ** 2, axis=1)


def rmse_global(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Root mean squared error across all frames and pixels.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Scalar RMSE value.
    """
    _check_shapes(Y_true, Y_pred)
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))


def r2_score(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²) across all frames and pixels.

    R² = 1 - SS_res / SS_tot, where SS_res is the sum of squared
    residuals and SS_tot is the total sum of squares around the mean.

    - R² = 1.0: perfect prediction.
    - R² = 0.0: predicts no better than the global mean.
    - R² < 0.0: worse than the mean.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Scalar R² value (computed over all elements, not per-pixel); for a
        constant ``Y_true``, 1.0 if the prediction is exact, else 0.0.
    """
    _check_shapes(Y_true, Y_pred)
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - Y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def error_map(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    image_shape: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Mean absolute error per pixel, reshaped to image grid.

    Shows where spatially the model makes the largest errors.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.
        image_shape: Pixel grid dimensions. Default ``(32, 32)``
            for Dräger PulmoVista 500.

    Returns:
        MAE per pixel, shape ``image_shape``.
    """
    _check_shapes(Y_true, Y_pred)
    return np.mean(np.abs(Y_true - Y_pred), axis=0).reshape(image_shape)


def correlation_per_frame(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Pearson correlation per frame (across pixels).

    Measures how well the spatial distribution is preserved in each frame.
    A value of 1.0 means the predicted image has the same spatial pattern
    as the ground truth (possibly scaled/shifted).

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Correlation per frame, shape ``(N_frames,)``. A constant frame has no
        correlation (0.0), unless the prediction is the same frame (1.0).
    """
    _check_shapes(Y_true, Y_pred)
    t = Y_true - Y_true.mean(axis=1, keepdims=True)
    p = Y_pred - Y_pred.mean(axis=1, keepdims=True)
    num = np.sum(t * p, axis=1)
    t_ss = np.sum(t**2, axis=1)
    p_ss = np.sum(p**2, axis=1)
    den = np.sqrt(t_ss * p_ss)
    corr = np.zeros_like(num, dtype=float)
    valid = den > 0
    corr[valid] = num[valid] / den[valid]
    same_constant = (t_ss == 0) & (p_ss == 0) & np.all(Y_true == Y_pred, axis=1)
    corr[same_constant] = 1.0
    return corr


def global_signal_correlation(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> float:
    """Pearson correlation of the global EIT signal (sum of all pixels).

    This is the most clinically relevant single metric: it tells whether
    the predicted tidal waveform matches the ground truth over time.

    The global EIT signal (sum of pixel impedance changes) is proportional
    to tidal volume as defined by the TREND consensus:
        Frerichs I et al. "Chest electrical impedance tomography
        examination, data analysis, terminology, clinical use and
        recommendations." *Thorax* 72(1), 2017, pp. 83-93.
        DOI: 10.1136/thoraxjnl-2016-208357

    Note: this metric evaluates waveform fidelity over time, not the
    per-breath tidal impedance variation (TIV) which is peak-to-trough.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Scalar Pearson correlation coefficient. A constant signal has no
        correlation (0.0), unless the prediction is the same signal (1.0).
    """
    _check_shapes(Y_true, Y_pred)
    g_true = Y_true.sum(axis=1)
    g_pred = Y_pred.sum(axis=1)
    if np.std(g_true) == 0 or np.std(g_pred) == 0:
        return 1.0 if np.array_equal(g_true, g_pred) else 0.0
    return float(np.corrcoef(g_true, g_pred)[0, 1])


def summary_metrics(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics in one call.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Dictionary with keys:
        - ``r2``: coefficient of determination (R²).
        - ``rmse``: root mean squared error (same units as pixels).
        - ``mse_mean``: mean MSE across all frames.
        - ``mse_std``: std of MSE across frames.
        - ``corr_spatial_mean``: mean per-frame spatial correlation.
        - ``corr_spatial_std``: std of per-frame spatial correlation.
        - ``corr_global``: Pearson r of global signal over time.
    """
    mse = mse_per_frame(Y_true, Y_pred)
    corr = correlation_per_frame(Y_true, Y_pred)
    return {
        "r2": r2_score(Y_true, Y_pred),
        "rmse": rmse_global(Y_true, Y_pred),
        "mse_mean": float(np.mean(mse)),
        "mse_std": float(np.std(mse)),
        "corr_spatial_mean": float(np.mean(corr)),
        "corr_spatial_std": float(np.std(corr)),
        "corr_global": global_signal_correlation(Y_true, Y_pred),
    }
