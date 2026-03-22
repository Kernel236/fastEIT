"""Utility helpers for Drager `.bin` parsing."""

from __future__ import annotations

import numpy as np

from fasteit.parsers.errors import InvalidSliceError

# ---------------------------------------------------------------------------
# Dräger PulmoVista 500 — sentinel constants
# ---------------------------------------------------------------------------
# Two distinct sentinel encodings in Dräger .bin files:
#   _BIT_SENTINELS  : uint32 bit-pattern 0xFF7FC99E — electrode not connected
#                     (hardware level; manifests as ~-3.4e38 when cast to float32)
#   _FLOAT_SENTINELS: float value -1000.0 — channel mode N/A or no breath
#                     calculated (software level)
_FLOAT_SENTINELS: tuple[float, ...] = (-1000.0,)
_BIT_SENTINELS: tuple[int, ...] = (0xFF7FC99E,)


def normalize_frame_slice(
    first_frame: int,
    max_frames: int | None,
    n_total_frames: int,
) -> tuple[int, int]:
    """Validate and normalize frame window into [start, stop)."""
    if first_frame < 0:
        raise InvalidSliceError("first_frame must be >= 0")
    if max_frames is not None and max_frames <= 0:
        raise InvalidSliceError("max_frames must be > 0 when provided")

    start = first_frame
    stop = n_total_frames if max_frames is None else first_frame + max_frames
    if n_total_frames > 0:
        stop = min(stop, n_total_frames)
    if n_total_frames > 0 and start >= n_total_frames:
        raise InvalidSliceError(
            f"first_frame={first_frame} out of bounds for n_total_frames={n_total_frames}"
        )
    return start, stop


def is_not_connected_sentinel(
    values: np.ndarray,
    bit_pattern_sentinels: tuple[int, ...],
) -> np.ndarray:
    """Return mask where float32 values match one raw uint32 sentinel pattern."""
    if not bit_pattern_sentinels:
        return np.zeros(values.shape, dtype=bool)

    as_u32 = values.view(np.uint32)
    mask = np.zeros(values.shape, dtype=bool)
    for sentinel in bit_pattern_sentinels:
        mask |= as_u32 == np.uint32(sentinel)
    return mask


def replace_no_data_sentinels(
    values: np.ndarray,
    float_sentinels: tuple[float, ...],
    bit_pattern_sentinels: tuple[int, ...],
) -> np.ndarray:
    """Return copy with vendor sentinel values replaced by NaN."""
    out = values.astype(np.float32, copy=True)
    out[is_not_connected_sentinel(out, bit_pattern_sentinels)] = np.nan
    for sentinel in float_sentinels:
        out[out == np.float32(sentinel)] = np.nan
    return out


def estimate_sampling_frequency_hz(timestamps_day_fraction: np.ndarray) -> float:
    """Derive fs from the validated frame timestamps.

    Args:
        timestamps_day_fraction: ``ts`` field values as fraction of day [0, 1),
            shape (N,).

    Returns:
        Sampling frequency in Hz.

    Raises:
        ValueError: if fewer than 2 timestamps are provided, or if the computed
            interval is non-positive (corrupt/identical timestamps).
    """
    ts = np.asarray(timestamps_day_fraction, dtype=np.float64)
    if ts.size < 2:
        raise ValueError("At least 2 timestamps required to estimate fs")
    seconds = np.unwrap(ts * 86400.0, period=86400.0)
    interval = seconds[-1] - seconds[0]
    if interval <= 0:
        raise ValueError("Non-positive timestamp interval — timestamps may be corrupt")
    return float((ts.size - 1) / interval)
