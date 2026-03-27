"""Evaluation metrics for EIT image reconstruction quality.

Compares predicted pixel arrays against ground-truth (.bin) to quantify
how well the Ridge model reproduces the Dräger reconstruction.
"""

from __future__ import annotations

import numpy as np


def mse_per_frame(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Mean squared error per frame.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        MSE per frame, shape ``(N_frames,)``.
    """
    return np.mean((Y_true - Y_pred) ** 2, axis=1)


def correlation_per_frame(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Pearson correlation per frame (across pixels).

    Measures how well the spatial distribution is preserved in each frame.
    A value of 1.0 means the predicted image has the same spatial pattern
    as the ground truth (possibly scaled/shifted).

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Correlation per frame, shape ``(N_frames,)``.
    """
    # Centre each frame
    t = Y_true - Y_true.mean(axis=1, keepdims=True)
    p = Y_pred - Y_pred.mean(axis=1, keepdims=True)
    num = np.sum(t * p, axis=1)
    den = np.sqrt(np.sum(t**2, axis=1) * np.sum(p**2, axis=1))
    # Avoid division by zero for constant frames
    den = np.where(den == 0, 1.0, den)
    return num / den


def global_signal_correlation(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> float:
    """Pearson correlation of the global EIT signal (sum of all pixels).

    This is the most clinically relevant single metric: it tells whether
    the predicted tidal waveform matches the ground truth over time.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Scalar Pearson correlation coefficient.
    """
    g_true = Y_true.sum(axis=1)
    g_pred = Y_pred.sum(axis=1)
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
        - ``mse_mean``: mean MSE across all frames.
        - ``mse_std``: std of MSE across frames.
        - ``corr_spatial_mean``: mean per-frame spatial correlation.
        - ``corr_spatial_std``: std of per-frame spatial correlation.
        - ``corr_global``: Pearson r of global signal over time.
    """
    mse = mse_per_frame(Y_true, Y_pred)
    corr = correlation_per_frame(Y_true, Y_pred)
    return {
        "mse_mean": float(np.mean(mse)),
        "mse_std": float(np.std(mse)),
        "corr_spatial_mean": float(np.mean(corr)),
        "corr_spatial_std": float(np.std(corr)),
        "corr_global": global_signal_correlation(Y_true, Y_pred),
    }
