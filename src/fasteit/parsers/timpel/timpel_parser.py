"""Timpel tabular EIT file parser.

Parses Timpel comma-separated export files (.csv / .txt) into
``ReconstructedFrameData``.

File format (1030 columns per row, no header, comma-delimited):
  cols  0–1023  32×32 reconstructed pixel impedance (row-major, arbitrary units)
  col  1024     airway pressure  (cmH₂O)
  col  1025     flow             (L/s)
  col  1026     volume           (L)
  col  1027     min_flag         1 = expiration trough detected, 0 = none
  col  1028     max_flag         1 = inspiration peak detected, 0 = none
  col  1029     qrs_flag         1 = QRS complex detected, 0 = none

Sentinel: −1000.0 → NaN (electrode disconnection or N/A channel).
Sampling frequency: 50 Hz (fixed by device firmware; no timestamps in file).

Format reference: eitprocessing timpel.py loader (Apache-2.0)
https://github.com/EIT-ALIVE/eitprocessing
Somhorst P et al., "eitprocessing", JOSS 2026;11(117):8179
DOI: 10.21105/joss.08179
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.detection import detect_vendor_from_tabular
from fasteit.parsers.errors import InvalidSliceError

from .timpel_dtypes import (
    TIMPEL_AIRWAY_PRESSURE_COL,
    TIMPEL_AUX_FIELDS,
    TIMPEL_COLUMN_COUNT,
    TIMPEL_DEFAULT_SAMPLE_FREQUENCY,
    TIMPEL_FRAME_DTYPE,
    TIMPEL_NAN_SENTINEL,
)


class TimpelTabularParser(BaseParser):
    """Parser for Timpel EIT tabular export files (.csv / .txt).

    Detects Timpel files by the characteristic 1030-column schema (no header).
    Supports partial loading via ``first_frame`` / ``max_frames``.

    Format reference: eitprocessing timpel.py loader (Apache-2.0).
    https://github.com/EIT-ALIVE/eitprocessing
    """

    def validate(self, path: Path) -> bool:
        """Return True if the file looks like a Timpel tabular export."""
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return False
        if path.suffix.lower() not in {".csv", ".txt", ".asc"}:
            return False
        try:
            vendor = detect_vendor_from_tabular(path)
        except ValueError:
            return False
        return vendor == "timpel"

    def parse(
        self,
        path: Path,
        first_frame: int = 0,
        max_frames: int | None = None,
    ) -> ReconstructedFrameData:
        """Parse Timpel CSV into ReconstructedFrameData.

        Args:
            path:        Path to the Timpel .csv / .txt file.
            first_frame: Zero-indexed frame to start from. Default 0.
            max_frames:  Maximum frames to load. None = all from first_frame.

        Returns:
            ReconstructedFrameData with:
            - frames: structured array with "ts" (seconds) and "pixels" (32×32)
            - aux_signals: dict with airway_pressure, flow, volume,
                           min_flag, max_flag, qrs_flag — each shape (N,)
            - fs: TIMPEL_DEFAULT_SAMPLE_FREQUENCY (50 Hz)

        Raises:
            ValueError:       Wrong column count or empty slice.
            InvalidSliceError: first_frame out of bounds or max_frames <= 0.
            OSError:           File cannot be read as text (encoding error).
        """
        path = Path(path)

        if first_frame < 0:
            raise InvalidSliceError("first_frame must be >= 0")
        if max_frames is not None and max_frames <= 0:
            raise InvalidSliceError("max_frames must be > 0 when provided")

        # ── 1. Load raw numeric matrix ─────────────────────────────────────
        # np.loadtxt skiprows=first_frame skips the first N rows of the file,
        # max_rows limits the number of rows read after that.
        try:
            raw: np.ndarray = np.loadtxt(
                str(path),
                dtype=np.float64,
                delimiter=",",
                skiprows=first_frame,
                max_rows=max_frames,
            )
        except UnicodeDecodeError as exc:
            raise OSError(
                f"'{path}' could not be decoded as a Timpel text file. "
                "Verify it is a valid Timpel export."
            ) from exc

        # ── 2. Validate schema ─────────────────────────────────────────────
        if raw.ndim == 1:
            if raw.size == 0:
                # Empty — skiprows exhausted all rows in the file
                raise InvalidSliceError(
                    f"first_frame={first_frame} is beyond the end of the file."
                )
            # Single row — np.loadtxt returns 1-D array; promote to 2-D
            raw = raw[np.newaxis, :]

        if raw.shape[0] == 0:
            raise InvalidSliceError(
                f"first_frame={first_frame} is beyond the end of the file."
            )

        if raw.shape[1] != TIMPEL_COLUMN_COUNT:
            raise ValueError(
                f"Expected {TIMPEL_COLUMN_COUNT} columns per row, "
                f"got {raw.shape[1]}. "
                "Verify this is a valid Timpel export file."
            )

        n_frames = raw.shape[0]

        # ── 3. Build synthetic timestamps (seconds from start) ─────────────
        # Timpel has no timestamp column. Synthetic time follows eitprocessing:
        #   time[i] = (first_frame + i) / fs
        # This preserves absolute frame indices even when loading a slice.
        fs = TIMPEL_DEFAULT_SAMPLE_FREQUENCY
        ts = (np.arange(n_frames) + first_frame) / fs

        # ── 4. Build structured frames array ───────────────────────────────
        frames = np.zeros(n_frames, dtype=TIMPEL_FRAME_DTYPE)
        frames["ts"] = ts

        # Reshape 1024 pixel columns to 32×32, replace NaN sentinel.
        # Use threshold comparison (< sentinel + 1.0) rather than exact equality
        # so the detection is robust to minor float32 rounding in future sentinel changes.
        pixels = raw[:, :TIMPEL_AIRWAY_PRESSURE_COL].astype(np.float32)
        pixels = pixels.reshape(n_frames, 32, 32)
        pixels = np.where(pixels < TIMPEL_NAN_SENTINEL + 1.0, np.nan, pixels)
        frames["pixels"] = pixels

        # ── 5. Build aux_signals dict ───────────────────────────────────────
        aux_signals: dict[str, np.ndarray] = {}
        for i, field_name in enumerate(TIMPEL_AUX_FIELDS):
            col = TIMPEL_AIRWAY_PRESSURE_COL + i
            values = raw[:, col].astype(np.float32)
            # Apply NaN sentinel to continuous channels (not to binary flags)
            if field_name not in {"min_flag", "max_flag", "qrs_flag"}:
                values = np.where(values < TIMPEL_NAN_SENTINEL + 1.0, np.nan, values)
            aux_signals[field_name] = values

        # ── 6. Assemble result ──────────────────────────────────────────────
        return ReconstructedFrameData(
            frames=frames,
            aux_signals=aux_signals,
            fs=fs,
            filename=str(path),
            file_format=path.suffix.lower().lstrip("."),
            metadata={
                "n_frames": n_frames,
                "first_frame_offset": first_frame,
            },
        )
