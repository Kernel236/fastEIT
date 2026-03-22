"""Drager `.bin` parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.errors import AmbiguousFormatError, UnsupportedFrameSizeError
from fasteit.parsers.detection import detect_bin_format_from_size

from .bin_utils import (
    _BIT_SENTINELS,
    _FLOAT_SENTINELS,
    estimate_sampling_frequency_hz,
    normalize_frame_slice,
    replace_no_data_sentinels,
)


class DragerBinParser(BaseParser):
    """Parser for Drager PulmoVista `.bin` files."""

    def __init__(self) -> None:
        self._float_sentinels = _FLOAT_SENTINELS
        self._bit_sentinels = _BIT_SENTINELS

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
            fs = 50.0
            warnings_list = [f"fs estimation failed ({e}); using default 50.0 Hz"]
        else:
            warnings_list = []

        # ── 6. Sanitize pixels: replace sentinel values with NaN ─────────────
        #TODO review if it's too slow!
        clean_pixels = np.zeros((n_frames_to_load, 32, 32), dtype=np.float32)
        for i in range(n_frames_to_load):
            clean_pixels[i] = replace_no_data_sentinels(
                mapped_frames["pixels"][i],
                self._float_sentinels,
                self._bit_sentinels,
            )

        # ── 7. Copy memmap to writable array, write clean pixels ──────────────
        # mapped_frames is read-only; copy to RAM before modifying.
        frames = mapped_frames[:]
        frames["pixels"] = clean_pixels

        # ── 8. Sanitize Medibus data if present (EXT format only) ─────────────
        if spec.medibus_fields is not None:
            clean_medibus = np.zeros(
                (n_frames_to_load, len(spec.medibus_fields)), dtype=np.float32
            )
            for i in range(n_frames_to_load):
                clean_medibus[i] = replace_no_data_sentinels(
                    frames["medibus_data"][i],
                    self._float_sentinels,
                    self._bit_sentinels,
                )
            frames["medibus_data"] = clean_medibus

        # ── 9. Build aux_signals dict: {signal_name → array shape (N,)} ──────
        aux_signals = None
        if spec.medibus_fields is not None:
            aux_signals = {
                field_name: frames["medibus_data"][:, field_idx]
                for field_idx, field_name in enumerate(spec.medibus_fields)
            }

        # ── 10. Assemble and return result ────────────────────────────────────
        result = ReconstructedFrameData(
            frames=frames,
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

