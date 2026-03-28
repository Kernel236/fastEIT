"""Evaluation metrics for EIT image reconstruction quality.

Compares predicted pixel arrays against ground-truth (.bin) to quantify
how well the Ridge model reproduces the Dräger reconstruction.

All functions are standalone (no model object needed) so they can be
used on saved predictions without reloading the fitted model.
"""

from __future__ import annotations

import numpy as np


def _validate_pair_shapes(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> None:
    """Validate that prediction/target arrays are compatible metrics inputs."""
    if Y_true.ndim != 2 or Y_pred.ndim != 2:
        raise ValueError(
            "Y_true and Y_pred must be 2D arrays with shape "
            "(N_frames, n_pixels). "
            f"Got ndim={Y_true.ndim} and ndim={Y_pred.ndim}."
        )
    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            "Y_true and Y_pred must have identical shape. "
            f"Got {Y_true.shape} vs {Y_pred.shape}."
        )


def mse_per_frame(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Mean squared error per frame.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        MSE per frame, shape ``(N_frames,)``.
    """
    _validate_pair_shapes(Y_true, Y_pred)
    return np.mean((Y_true - Y_pred) ** 2, axis=1)


def rmse_global(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Root mean squared error across all frames and pixels.

    Unlike MSE, RMSE is in the same units as the pixel values,
    making it directly comparable to the signal range.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Scalar RMSE value.
    """
    _validate_pair_shapes(Y_true, Y_pred)
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))


def r2_score(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²) across all frames and pixels.

    R² = 1 − SS_res / SS_tot, where SS_res is the sum of squared
    residuals and SS_tot is the total sum of squares around the mean.

    - R² = 1.0: perfect prediction.
    - R² = 0.0: predicts no better than the global mean.
    - R² < 0.0: worse than the mean.

    Note:
        This computes a **pooled** R² over all elements (frames × pixels),
        using the grand mean.  This differs from sklearn's
        ``r2_score(multioutput='uniform_average')`` which averages per-pixel
        R² values.  For baseline-subtracted EIT data the two are similar,
        but the pooled version is dominated by the tidal waveform.
        Use ``correlation_per_frame`` to assess spatial fidelity separately.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Scalar R² value (computed over all elements, not per-pixel).
    """
    _validate_pair_shapes(Y_true, Y_pred)
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
    High-error regions are typically near the heart (cardiac artifact)
    and at the edges of the lung fields (low SNR).

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.
        image_shape: Pixel grid dimensions. Default ``(32, 32)``
            for Dräger PulmoVista 500.

    Returns:
        MAE per pixel, shape ``image_shape``.
    """
    _validate_pair_shapes(Y_true, Y_pred)
    n_pixels = int(np.prod(image_shape))
    if Y_true.shape[1] != n_pixels:
        raise ValueError(
            "image_shape is incompatible with number of pixels. "
            f"Got n_pixels={Y_true.shape[1]} but image_shape={image_shape} "
            f"(prod={n_pixels})."
        )
    mae = np.mean(np.abs(Y_true - Y_pred), axis=0)
    return mae.reshape(image_shape)


def correlation_per_frame(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Pearson correlation per frame (across pixels).

    Measures how well the spatial distribution is preserved in each frame.
    A value of 1.0 means the predicted image has the same spatial pattern
    as the ground truth (possibly scaled/shifted).

    Spatial correlation is used as EIT image quality metric in:
        Scaramuzzo G et al. "Electrical Impedance Tomography:
        a technical guide for clinicians." *Crit Care* 28, 413 (2024).
        DOI: 10.1186/s13054-024-05173-x

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Correlation per frame, shape ``(N_frames,)``.
    """
    _validate_pair_shapes(Y_true, Y_pred)
    # Centre each frame
    t = Y_true - Y_true.mean(axis=1, keepdims=True)
    p = Y_pred - Y_pred.mean(axis=1, keepdims=True)
    num = np.sum(t * p, axis=1)
    t_ss = np.sum(t**2, axis=1)
    p_ss = np.sum(p**2, axis=1)
    den = np.sqrt(t_ss * p_ss)

    corr = np.zeros_like(num, dtype=float)
    valid = den > 0
    corr[valid] = num[valid] / den[valid]

    # Constant-frame handling:
    # - both constant -> perfect match if equal constants, else 0
    # - only one constant -> 0 (undefined Pearson, treated as no correlation)
    both_const = (t_ss == 0) & (p_ss == 0)
    if np.any(both_const):
        equal_const = np.all(
            Y_true[both_const] == Y_pred[both_const],
            axis=1,
        )
        corr[both_const] = np.where(equal_const, 1.0, 0.0)

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
        recommendations." *Thorax* 72(1), 2017, pp. 83–93.
        DOI: 10.1136/thoraxjnl-2016-208357

    Note: this metric evaluates waveform fidelity over time, not the
    per-breath tidal impedance variation (TIV) which is peak-to-trough.

    Args:
        Y_true: Ground truth, shape ``(N_frames, n_pixels)``.
        Y_pred: Prediction, shape ``(N_frames, n_pixels)``.

    Returns:
        Scalar Pearson correlation coefficient.
    """
    _validate_pair_shapes(Y_true, Y_pred)
    g_true = Y_true.sum(axis=1)
    g_pred = Y_pred.sum(axis=1)

    gt_std = float(np.std(g_true))
    gp_std = float(np.std(g_pred))

    # Pearson is undefined for constant signals; use deterministic fallback.
    if gt_std == 0 and gp_std == 0:
        return 1.0 if np.all(g_true == g_pred) else 0.0
    if gt_std == 0 or gp_std == 0:
        return 0.0

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
