"""Post-processing of the reconstructed images.

The Dräger images are low-pass filtered in time, with no delay: the ridge reconstructs every
frame on its own and keeps the heart beat and the noise. A zero-phase low-pass at 50/min, the
low-pass cutoff of the Dräger software, brings the reconstruction to the same smoothing.

Source: comparison with paired .bin recordings of new patients (contiguous frames): the R² of
the changes rose from 0.81-0.93 to 0.98-0.99 with the zero-phase filter, and fell with a causal
one (which delays the signal).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

LOWPASS_BPM = 50  # cutoff, per minute
LOWPASS_ORDER = 2  # Butterworth order (applied twice: forward and backward)


def lowpass(
    images: np.ndarray, fs: float, cutoff_bpm: float = LOWPASS_BPM
) -> np.ndarray:
    """Zero-phase Butterworth low-pass along time (axis 0).

    Args:
        images: Reconstructed frames, shape ``(N_frames, ...)``, contiguous in time.
        fs: Frame rate in Hz.
        cutoff_bpm: Cutoff frequency, per minute.

    Returns:
        Filtered frames, same shape.

    Raises:
        ValueError: If the recording is too short for the filter's edge padding.
    """
    sos = signal.butter(LOWPASS_ORDER, cutoff_bpm / 60.0, fs=fs, output="sos")
    # Edge padding of sosfiltfilt (scipy's default): the recording must be longer.
    zeros = min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    padding = 3 * (2 * len(sos) + 1 - zeros)
    if images.shape[0] <= padding:
        raise ValueError(
            f"Recording too short for the low-pass: {images.shape[0]} frames, "
            f"more than {padding} needed."
        )
    return signal.sosfiltfilt(sos, images, axis=0)
