"""Drager `.bin` parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.detection import detect_bin_format_from_size
from fasteit.parsers.errors import AmbiguousFormatError, UnsupportedFrameSizeError

from .bin_utils import (
    estimate_sampling_frequency_hz,
    normalize_frame_slice,
)

# Default sampling frequency used when fs cannot be estimated from timestamps.
# PulmoVista 500 acquisition rate is typically 20 Hz (standard) or 50 Hz
# (high-speed mode). 50 Hz is the safe upper bound; recordings with a
# PressurePod always run at 50 Hz (device specification).
_DEFAULT_FS_HZ: float = 50.0


class DragerBinParser(BaseParser):
    """Parser for Dräger PulmoVista 500 `.bin` files.

    Supports two frame layouts identified automatically by file size:
    - BASE (4358 bytes/frame): 32×32 pixels + 52 Medibus channels
    - EXT  (4382 bytes/frame): 32×32 pixels + 58 Medibus channels
      (PressurePod variant with esophageal/transpulmonary pressure fields)

    New frame sizes can be added by registering a ``FormatSpec`` in
    ``parsers/bin_formats.py`` without modifying this parser.

    Format sources:
    - Reverse-engineered from PulmoVista 500 binary output
    - Cross-referenced with eitprocessing (Apache-2.0):
      https://github.com/EIT-ALIVE/eitprocessing
    """

    def validate(self, path: Path) -> bool:
        """Return True if file exists, is non-empty, and has a known frame size."""
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            detect_bin_format_from_size(path)
        except (UnsupportedFrameSizeError, AmbiguousFormatError):
            return False
        return True

    def parse(
        self,
        path: Path,
        first_frame: int = 0,
        max_frames: int | None = None,
    ) -> ReconstructedFrameData:
        """Parse `.bin` file into ReconstructedFrameData.

        Args:
            path: Path to the .bin file.
            first_frame: Zero-indexed frame to start from. Default 0.
            max_frames: Maximum number of frames to load. If None, load all
                        frames from first_frame to end of file.

        Returns:
            ReconstructedFrameData with frames, timestamps, pixels,
            aux_signals (Medibus dict), fs, and metadata.

        Raises:
            UnsupportedFrameSizeError: If file size is not a multiple of a known frame size.
            AmbiguousFormatError: If file size is a multiple of >1 frame sizes (rare).
            InvalidSliceError: If first_frame/max_frames are invalid.
        """
        path = Path(path)

        # ── 1. Detect frame format from file size ─────────────────────────────
        spec = detect_bin_format_from_size(path)

        # ── 2. Count total frames (no file read, just filesystem metadata) ────
        file_size = path.stat().st_size
        n_total_frames = file_size // spec.frame_size_bytes

        # ── 3. Resolve requested window into (start, stop) indices ───────────
        start, stop = normalize_frame_slice(
            first_frame=first_frame,
            max_frames=max_frames,
            n_total_frames=n_total_frames,
        )
        n_frames_to_load = stop - start

        # ── 4. Memory-map only the requested slice ────────────────────────────
        # offset jumps directly to byte position of frame `start` — frames
        mapped_frames = np.memmap(
            path,
            dtype=spec.dtype,
            mode="r",
            offset=start * spec.frame_size_bytes,
            shape=(n_frames_to_load,),
        )

        # ── 5. Estimate sampling frequency from timestamps ────────────────────
        try:
            fs = estimate_sampling_frequency_hz(mapped_frames["ts"])
        except ValueError as e:
            fs = _DEFAULT_FS_HZ
            warnings_list = [
                f"fs estimation failed ({e}); using default {_DEFAULT_FS_HZ} Hz"
            ]
        else:
            warnings_list = []

        # ── 6. Build aux_signals dict: {signal_name → array shape (N,)} ──────
        aux_signals = None
        if spec.medibus_fields is not None:
            aux_signals = {
                field_name: mapped_frames["medibus_data"][:, field_idx]
                for field_idx, field_name in enumerate(spec.medibus_fields)
            }

        # ── 10. Assemble and return result ────────────────────────────────────
        result = ReconstructedFrameData(
            frames=mapped_frames,
            aux_signals=aux_signals,
            fs=fs,
            filename=str(path),
            file_format="bin",
            metadata={
                "n_total_frames": n_total_frames,
                "n_loaded_frames": n_frames_to_load,
                "frame_format": spec.name,
                "has_pressure_pod": spec.has_pressure_pod_fields,
            },
        )

        if warnings_list:
            result.metadata["warnings"] = warnings_list

        return result
