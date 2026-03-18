"""BinData: container for parsed .bin file data (reconstructed 32×32 images)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base_data import BaseData


@dataclass
class BinData(BaseData):
    """Container for data from a Dräger PulmoVista .bin file.

    The .bin file contains 32×32 impedance images already reconstructed by the
    PulmoVista hardware — these are NOT raw electrode voltages. Each frame is a
    structured numpy record with timestamps, pixel values, event fields, and an
    optional Medibus block.

    Attributes:
        frames:  Structured numpy array, shape (N,), dtype FRAME_BASE_DTYPE or
                 FRAME_EXT_DTYPE (defined in fasteit.dtypes, Task 1.2.1).
        medibus: Dict with Medibus signals if frame is extended type, else None.
    """

    frames: np.ndarray = None
    medibus: dict | None = None

    def __post_init__(self) -> None:
        if self.frames is not None:
            self.n_frames = len(self.frames)
            self.duration = self.n_frames / self.fs

    # ── Direct field accessors ────────────────────────────────────────────

    @property
    def timestamps(self) -> np.ndarray:
        """Timestamp ts1 per frame. Shape (N,)."""
        return self.frames["ts1"]

    @property
    def pixels(self) -> np.ndarray:
        """Reconstructed 32×32 images for all frames. Shape (N, 32, 32)."""
        return self.frames["pixels"]

    @property
    def event_texts(self) -> np.ndarray:
        """Event text strings per frame. Shape (N,) of bytes."""
        return self.frames["event_text"]

    # ── Derived signals ───────────────────────────────────────────────────
    # These operate on ALL pixels. The clinically meaningful signal should use
    # only pixels within the lung_mask, computed in the preprocessing layer.

    @property
    def global_signal(self) -> np.ndarray:
        """Sum of all pixels per frame (raw, no lung mask). Shape (N,)."""
        return self.frames["pixels"].sum(axis=(1, 2))

    @property
    def roi_signals(self) -> np.ndarray:
        """4 ventro-dorsal ROI signals (equal horizontal strips). Shape (N, 4).

        ROI 0 = ventral, ROI 3 = dorsal. Each ROI covers 8 rows of the 32×32 grid.

        Note: these fixed ROIs are one possible definition. Configurable ROIs
        are supported in the preprocessing layer (Task 4.8).
        """
        p = self.frames["pixels"]
        return np.column_stack(
            [
                p[:, 0:8, :].sum(axis=(1, 2)),  # ROI 0 — ventrale
                p[:, 8:16, :].sum(axis=(1, 2)),  # ROI 1 — mid-ventrale
                p[:, 16:24, :].sum(axis=(1, 2)),  # ROI 2 — mid-dorsale
                p[:, 24:32, :].sum(axis=(1, 2)),  # ROI 3 — dorsale
            ]
        )

    def roi_signal(self, roi: int) -> np.ndarray:
        """Signal for a single ROI (0 = ventral … 3 = dorsal). Shape (N,).

        Args:
            roi: ROI index in range 0–3.

        Raises:
            ValueError: if roi is not in range 0–3.
        """
        if roi not in range(4):
            raise ValueError(f"ROI deve essere 0-3, ricevuto {roi}")
        start = roi * 8
        return self.frames["pixels"][:, start : start + 8, :].sum(axis=(1, 2))
