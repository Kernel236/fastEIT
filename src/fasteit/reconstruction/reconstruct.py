"""From an ``.eit`` recording to ``ReconstructedFrameData``, with a trained model.

The images go through the rest of fastEIT unchanged (``preprocess``, indices, export), like
the images of a ``.bin``: zero at the bottom of the recording, low-pass at 50/min with zero
phase by default (the Dräger setting; ``None`` turns it off for a monitor set otherwise).

Without ``like``, frame times and the named ventilator signals come from the ``.eit`` itself
(the PressurePod is not in the ``.eit``). ``like``, the ``.bin`` of the same acquisition when
there is one, gives the frame times, the device events and timing errors, and all its signals:
in a comparison with the ``.bin``, only the images differ. Its breath marks (``min_max_flag``),
computed by Dräger on its own images, are not copied.
"""

from __future__ import annotations

import numpy as np

from fasteit.models import RawImpedanceData, ReconstructedFrameData
from fasteit.parsers.draeger.eit.eit_dtypes import EIT_MEDIBUS_CHANNELS
from fasteit.reconstruction.model import EITReconstructionModel
from fasteit.reconstruction.postprocess import LOWPASS_BPM, lowpass

IMAGE_SHAPE = (32, 32)
# Frame fields copied from ``like``: device events and timing errors.
COPIED_FIELDS = ("event_marker", "event_text", "timing_error")


def reconstruct(
    raw: RawImpedanceData,
    model: EITReconstructionModel,
    lowpass_bpm: float | None = LOWPASS_BPM,
    like: ReconstructedFrameData | None = None,
) -> ReconstructedFrameData:
    """Reconstruct the images of a whole ``.eit`` recording.

    Args:
        raw: Parsed ``.eit`` (``load_data``).
        model: Trained model (``fit_model`` or ``EITReconstructionModel.load``).
        lowpass_bpm: Low-pass cutoff per minute, or ``None`` for no filter.
        like: The ``.bin`` of the same acquisition, if any (frame times, events,
            ventilator signals).

    Returns:
        Images ``(N, 32, 32)`` with their frame times, ready for ``preprocess()``.

    Raises:
        ValueError: If ``like`` has a different number of frames, or the recording is
            too short for the low-pass.
    """
    pixels = model.predict(np.asarray(raw.measurements, dtype=float))
    if lowpass_bpm is not None:
        pixels = lowpass(pixels, raw.fs, lowpass_bpm)
    n = len(pixels)
    if like is not None and like.n_frames != n:
        raise ValueError(f"like has {like.n_frames} frames, the .eit has {n}.")

    copied = [
        f for f in COPIED_FIELDS if like is not None and f in like.frames.dtype.names
    ]
    dtype = [("ts", "<f8"), ("pixels", "<f4", IMAGE_SHAPE)]
    dtype += [(f, like.frames.dtype[f]) for f in copied]
    frames = np.zeros(n, dtype=dtype)
    frames["pixels"] = pixels.reshape(n, *IMAGE_SHAPE)
    frames["ts"] = like.timestamps if like is not None else raw.aux_signals["timestamp"]
    for f in copied:
        frames[f] = like.frames[f]

    if like is not None:
        aux_signals = like.aux_signals
    else:
        eit_signals = raw.aux_signals or {}
        aux_signals = {
            k: eit_signals[k] for k in EIT_MEDIBUS_CHANNELS if k in eit_signals
        }
    return ReconstructedFrameData(
        frames=frames,
        aux_signals=aux_signals or None,
        fs=raw.fs,
        filename=raw.filename,
        file_format="eit-ml",
        vendor="draeger",
        metadata={"lowpass_bpm": lowpass_bpm, "alpha": model.ridge.alpha},
    )
