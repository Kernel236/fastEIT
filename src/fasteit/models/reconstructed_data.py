"""ReconstructedFrameData: container for frame-wise reconstructed EIT data.

Produced by: DragerBinParser (.bin), TimpelTabularParser (.txt/.csv).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base_data import BaseData


@dataclass
class ReconstructedFrameData(BaseData):
    """Container for frame-wise reconstructed EIT data with synchronized signals.

    Stores vendor-reconstructed 32×32 pixel images and any continuous signals
    that are frame-synchronised in the source file (e.g. Medibus waveforms in
    Dräger .bin, device channels in Timpel .txt).

    ``aux_signals`` is a vendor-agnostic dict of named continuous signals that
    are frame-aligned with ``frames``. For Dräger .bin this is the Medibus dict;
    for Timpel .txt this will hold the 5 device channels.

    Attributes:
        frames:      Structured numpy array, shape (N,). Required fields: ``ts``
                     (float64, timestamp), ``pixels`` (float32, 32×32 image).
        aux_signals: Dict mapping signal name → 1-D array of length N, or None
                     if the source file carries no auxiliary channels.
    """

    frames: np.ndarray | None = None
    aux_signals: dict | None = None

    def __post_init__(self) -> None:
        if self.frames is not None:
            self.n_frames = len(self.frames)
            self.duration = self.n_frames / self.fs if self.fs else 0.0

    # ── Internal guard ─────────────────────────────────────────────────────

    def _require_frames(self) -> np.ndarray:
        if self.frames is None:
            raise AttributeError(
                "No frames loaded. Instantiate ReconstructedFrameData with frames=<array>."
            )
        return self.frames

    # ── Direct field accessors ────────────────────────────────────────────

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamp per frame as fraction of day (0.0–1.0). Shape (N,).
        """
        return self._require_frames()["ts"]

    @property
    def pixels(self) -> np.ndarray:
        """Reconstructed 32×32 images for all frames. Shape (N, 32, 32)."""
        return self._require_frames()["pixels"]

    @property
    def min_max_flags(self) -> np.ndarray:
        """Breath phase marker per frame. Shape (N,). Dräger .bin only.

        +1 = inspiration peak, -1 = expiration trough, 0 = neither.
        """
        frames = self._require_frames()
        if "min_max_flag" not in frames.dtype.names:
            raise AttributeError(
                "min_max_flag is not available for this recording. "
                "This field is only present in Dräger .bin files."
            )
        return frames["min_max_flag"]

    @property
    def event_markers(self) -> np.ndarray:
        """Event counter per frame. Shape (N,). Dräger .bin only.
        """
        frames = self._require_frames()
        if "event_marker" not in frames.dtype.names:
            raise AttributeError(
                "event_marker is not available for this recording. "
                "This field is only present in Dräger .bin files."
            )
        return frames["event_marker"]

    @property
    def event_texts(self) -> np.ndarray:
        """Event text strings per frame. Shape (N,) of bytes. Dräger .bin only.
        """
        frames = self._require_frames()
        if "event_text" not in frames.dtype.names:
            raise AttributeError(
                "event_text is not available for this recording. "
                "This field is only present in Dräger .bin files."
            )
        return frames["event_text"]

    # ── Derived signals ───────────────────────────────────────────────────

    @property
    def global_signal(self) -> np.ndarray:
        """Sum of all pixels per frame (no lung mask applied). Shape (N,)."""
        return self._require_frames()["pixels"].sum(axis=(1, 2))

    @property
    def roi_signals(self) -> np.ndarray:
        """4 ventro-dorsal ROI signals (equal horizontal strips). Shape (N, 4).

        ROI 0 = ventral (rows 0–7), ROI 3 = dorsal (rows 24–31).
        """
        p = self._require_frames()["pixels"]
        return np.column_stack(
            [
                p[:, 0:8, :].sum(axis=(1, 2)),   # ROI 0 — ventral
                p[:, 8:16, :].sum(axis=(1, 2)),  # ROI 1 — mid-ventral
                p[:, 16:24, :].sum(axis=(1, 2)), # ROI 2 — mid-dorsal
                p[:, 24:32, :].sum(axis=(1, 2)), # ROI 3 — dorsal
            ]
        )

    def roi_signal(self, roi: int) -> np.ndarray:
        """Signal for a single ROI (0 = ventral … 3 = dorsal). Shape (N,).

        Args:
            roi: ROI index in range 0–3.
        """
        if roi < 0 or roi > 3:
            raise ValueError(f"roi must be 0–3, got {roi}")
        start = roi * 8
        return self._require_frames()["pixels"][:, start : start + 8, :].sum(axis=(1, 2))
