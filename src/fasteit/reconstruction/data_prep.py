"""Data preparation for paired .eit/.bin reconstruction training.

Each recording is normalised on its own reference, taken at the bottom of the recording: the
``VALLEY_FRACTION`` of its frames where the median relative change of the channels is lowest
(the lowest end-expirations of the breathing curve, found from X alone). This is where the
Dräger images have their zero: their raw global signal is almost never below 0, and reaches 0
at the lowest end-expiratory level of the recording, wherever it is (late in a decremental PEEP
trial). The reference of X and of Y is the mean at those same frames:

- the input X (transimpedances) by the relative change ``(x - ref) / ref`` (normalised
  difference). Contact, thorax size and gain multiply a whole channel, and the division
  cancels that factor, so one map fits different patients. A channel whose reference is too
  small or has the unusual sign would be amplified or inverted: it is set to 0 (a bad
  electrode, as the Dräger matrices with one electrode left out). Typical reference and
  usual sign come from the training recordings only;
- the target Y (pixels) by subtraction, ``y - ref``: the clinical indices are changes
  (TIV, dEELI) or ratios of changes, so the offset does not matter.

Source: leave-one-patient-out comparisons on paired recordings (subtraction, relative change,
relative change with bad channels set to 0; reference in the first frames or at the bottom of the
recording), in notebooks kept private with the results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fasteit.parsers.loader import load_data

# Fraction of the frames of a recording, the lowest of the breathing curve, averaged for the
# reference; at least MIN_REFERENCE_FRAMES (sampled recordings can be short).
VALLEY_FRACTION = 0.02
MIN_REFERENCE_FRAMES = 10
# A channel is bad if its |reference| is below this fraction of the typical one.
BAD_REF_FRAC = 0.2
# A recording is left out of the training data if the RMS of its normalised pixels is above
# this many times the median of the recordings: Dräger images far beyond breathing (reference
# taken in a "low" state, bottom on a disconnection, huge values in a few pixels). Least squares
# weighs a recording by the square of its values, so a few of them would drive the whole fit.
# Natural gap in the data: 4.2 to 317 times (old dataset), 7.1 to 15 times (new study).
TRAINING_RMS_MAX = 10.0


def load_paired(
    eit_path: str | Path,
    bin_path: str | Path,
    input_mode: str = "vv",
    max_frame_diff: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a paired .eit/.bin recording and return aligned arrays.

    Args:
        eit_path: Path to the .eit file.
        bin_path: Path to the corresponding .bin file.
        input_mode: Feature set for input X:
            - ``"vv"``: calibrated transimpedances (N, 208) - default.
            - ``"raw"``: concatenated [trans_A, trans_B] (N, 416).
        max_frame_diff: Files this many frames apart are trimmed to the shorter one
            (one file of a pair can have one frame more).

    Returns:
        Tuple ``(X, Y)``: X of shape ``(N, 208)`` or ``(N, 416)``, Y the flattened
        pixels, shape ``(N, 1024)``.

    Raises:
        ValueError: If the frame counts differ by more than ``max_frame_diff`` or
            input_mode is unknown.
    """
    eit_data = load_data(eit_path)
    bin_data = load_data(bin_path)

    if input_mode == "vv":
        X = eit_data.measurements  # (N, 208)
    elif input_mode == "raw":
        aux = eit_data.aux_signals
        X = np.hstack([aux["trans_A"], aux["trans_B"]])  # (N, 416)
    else:
        raise ValueError(f"Unknown input_mode '{input_mode}'. Use 'vv' or 'raw'.")

    Y = bin_data.pixels.reshape(bin_data.n_frames, -1)  # (N, 1024)

    if abs(X.shape[0] - Y.shape[0]) > max_frame_diff:
        raise ValueError(
            f"Frame count mismatch: .eit has {X.shape[0]}, "
            f".bin has {Y.shape[0]}. Files may not be from the same recording."
        )
    n = min(X.shape[0], Y.shape[0])
    return X[:n], Y[:n]


def reference_frames(
    x: np.ndarray,
    low: float = VALLEY_FRACTION,
    min_frames: int = MIN_REFERENCE_FRAMES,
) -> np.ndarray:
    """Lowest end-expiratory frames of a recording, from its transimpedances X only.

    The breathing curve is the median over the channels of the relative change from the mean
    of the recording: the channels grow together in magnitude with inspiration, and the median
    ignores the few with a reference close to 0, which explode.

    Args:
        x: One whole recording, shape ``(N_frames, channels)``.
        low: Fraction of the frames taken, the lowest of the curve.
        min_frames: Smallest number of frames taken.

    Returns:
        Sorted frame indices.
    """
    m = x.mean(axis=0)
    curve = np.median((x - m) / m, axis=1)
    n = min(len(x), max(min_frames, int(low * len(x))))
    return np.sort(np.argsort(curve)[:n])


def reference(arr: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """Mean of a recording at its reference frames.

    Args:
        arr: One recording, shape ``(N_frames, features)``.
        frames: Reference frames (``reference_frames`` of its X).

    Returns:
        Reference per feature, shape ``(features,)``.
    """
    return arr[frames].mean(axis=0)


def normalize(arr: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the reference of a recording (used for the pixels Y).

    Args:
        arr: One recording, shape ``(N_frames, features)``.
        frames: Reference frames, found on the X of the same recording.

    Returns:
        Tuple ``(delta, ref)``: the recording minus its reference, and the reference.
    """
    ref = reference(arr, frames)
    return arr - ref, ref


def channel_stats(refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Typical |reference| and usual sign of each channel, from training recordings.

    Args:
        refs: References of the training recordings, shape ``(n_recordings, channels)``.

    Returns:
        Tuple ``(typical_abs, usual_sign)``: median |reference| and majority sign (+1 on
        a tie) of each channel.
    """
    typical_abs = np.median(np.abs(refs), axis=0)
    usual_sign = np.where(np.sign(refs).sum(axis=0) >= 0, 1.0, -1.0)
    return typical_abs, usual_sign


def bad_channels(
    ref: np.ndarray,
    typical_abs: np.ndarray,
    usual_sign: np.ndarray,
    frac: float = BAD_REF_FRAC,
) -> np.ndarray:
    """Channels of a recording whose reference is too small or has the unusual sign.

    Args:
        ref: Reference of the recording, shape ``(channels,)``.
        typical_abs: Typical |reference| per channel (``channel_stats``).
        usual_sign: Usual sign per channel (``channel_stats``).
        frac: Bad if ``|ref| < frac x typical_abs``.

    Returns:
        Boolean mask, shape ``(channels,)``.
    """
    return (np.abs(ref) < frac * typical_abs) | (np.sign(ref) != usual_sign)


def normalize_relative(
    x: np.ndarray,
    frames: np.ndarray,
    typical_abs: np.ndarray,
    usual_sign: np.ndarray,
) -> np.ndarray:
    """Relative change ``(x - ref) / ref`` of a recording, bad channels set to 0 (used for X).

    Args:
        x: One recording, shape ``(N_frames, channels)``.
        frames: Reference frames (``reference_frames(x)``).
        typical_abs: Typical |reference| per channel, from the training recordings.
        usual_sign: Usual sign per channel, from the training recordings.

    Returns:
        Normalised recording, shape ``(N_frames, channels)``.
    """
    ref = reference(x, frames)
    bad = bad_channels(ref, typical_abs, usual_sign)
    safe_ref = np.where(bad, 1.0, ref)
    out = (x - ref) / safe_ref
    out[:, bad] = 0.0
    return out


def pixel_rms(y: np.ndarray) -> float:
    """Root mean square of all the pixels of a normalised recording.

    It sees anomalies that cancel in the global signal (huge values of opposite sign in a
    few pixels), which the swing of the global signal does not.

    Args:
        y: Normalised pixels of one recording, shape ``(N_frames, n_pixels)``.

    Returns:
        RMS over frames and pixels.
    """
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
