"""Data preparation for paired .eit/.bin reconstruction training.

Loads paired recordings, normalises per-file (baseline subtraction),
and returns concatenated arrays ready for model training.

Normalisation: subtract the mean of the first ``n_ref`` frames from each
recording independently.  This removes the absolute impedance baseline
(which varies across patients and after device recalibration) and leaves
only the differential signal — tidal variations, EELI trends, etc.
Per-file normalisation is required so that recordings from different
patients or sessions can be safely concatenated for training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fasteit.parsers.loader import load_data


def load_paired(
    eit_path: str | Path,
    bin_path: str | Path,
    input_mode: str = "vv",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a paired .eit/.bin recording and return aligned arrays.

    Args:
        eit_path: Path to the .eit file.
        bin_path: Path to the corresponding .bin file.
        input_mode: Feature set for input X:
            - ``"vv"``: calibrated transimpedances (N, 208) — default.
            - ``"raw"``: concatenated [trans_A, trans_B] (N, 416).

    Returns:
        Tuple ``(X, Y)`` where:
        - X: input features, shape ``(N_frames, 208)`` or ``(N_frames, 416)``.
        - Y: flattened pixel targets, shape ``(N_frames, 1024)``.

    Raises:
        ValueError: If frame counts do not match or input_mode is unknown.
    """
    eit_data = load_data(eit_path)
    bin_data = load_data(bin_path)

    if input_mode == "vv":
        X = eit_data.measurements  # (N, 208)
    elif input_mode == "raw":
        trans_A = eit_data.aux_signals["trans_A"]  # (N, 208)
        trans_B = eit_data.aux_signals["trans_B"]  # (N, 208)
        X = np.hstack([trans_A, trans_B])  # (N, 416)
    else:
        raise ValueError(
            f"Unknown input_mode '{input_mode}'. Use 'vv' or 'raw'."
        )

    Y = bin_data.pixels.reshape(bin_data.n_frames, -1)  # (N, 1024)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Frame count mismatch: .eit has {X.shape[0]}, "
            f".bin has {Y.shape[0]}. Files may not be from the same recording."
        )

    return X, Y


def normalize(
    arr: np.ndarray,
    n_ref: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract baseline (mean of first ``n_ref`` frames) from array.

    Args:
        arr: Input array, shape ``(N_frames, features)``.
        n_ref: Number of frames to average for baseline. Default 50.

    Returns:
        Tuple ``(delta, ref)`` where:
        - delta: ``(N_frames, features)`` baseline-subtracted array.
        - ref: ``(features,)`` baseline vector (kept for inference).
    """
    ref = arr[:n_ref].mean(axis=0)
    return arr - ref, ref


def prepare_dataset(
    pairs: list[tuple[str | Path, str | Path]],
    n_ref: int = 50,
    input_mode: str = "vv",
) -> tuple[np.ndarray, np.ndarray]:
    """Load multiple paired recordings, normalise per-file, concatenate.

    Each recording is normalised independently (baseline = mean of first
    ``n_ref`` frames) before concatenation.  This is safe regardless of
    device recalibration or patient differences.

    Args:
        pairs: List of ``(eit_path, bin_path)`` tuples.
        n_ref: Frames for baseline normalisation per recording.
        input_mode: ``"vv"`` (208 features) or ``"raw"`` (416 features).

    Returns:
        Tuple ``(X, Y)`` where:
        - X: ``(total_frames, n_features)`` normalised input.
        - Y: ``(total_frames, 1024)`` normalised pixel targets.
    """
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for eit_path, bin_path in pairs:
        X, Y = load_paired(eit_path, bin_path, input_mode=input_mode)
        delta_x, _ = normalize(X, n_ref=n_ref)
        delta_y, _ = normalize(Y, n_ref=n_ref)
        all_x.append(delta_x)
        all_y.append(delta_y)

    return np.concatenate(all_x), np.concatenate(all_y)
