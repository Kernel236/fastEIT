"""Dräger `.eit` parser: the raw transimpedances of the PulmoVista 500 file."""

# File layout (reverse-engineered from PulmoVista 500 output, cross-referenced with
# EIDORS read_draeger_header/read_draeger_file, format study only):
#   bytes 0-11                 preamble, 3 x int32 LE: format_version (always 51),
#                              sep_offset (file-specific), unknown
#   bytes 12 to sep_offset-1   ASCII header (latin-1, "key: value\r\n" lines)
#   sep_offset, 8 bytes        separator b"**\r\n\r\n\r\n"
#   from sep_offset+8          binary frames, 5495 bytes each (eit_dtypes)

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from fasteit.models.raw_impedance_data import RawImpedanceData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.detection import detect_vendor_from_eit_header
from fasteit.parsers.header_formats import get_eit_specs

from .eit_dtypes import FRAME_EIT_DTYPE, PREAMBLE_DTYPE, PREAMBLE_N_FIELDS
from .eit_utils import (
    CURRENT_COUNTS_PER_AMPERE,
    PRIMARY_TRANSIMPEDANCE_SCALE,
    REFERENCE_TRANSIMPEDANCE_SCALE,
    SEPARATOR,
    VOLTAGE_COUNTS_PER_VOLT,
    parse_eit_header,
)


class DragerEitParser(BaseParser):
    """Parser for Dräger PulmoVista 500 `.eit` files.

    Produces a ``RawImpedanceData`` container with the raw transimpedance
    measurements (208 values per frame) extracted from the binary section
    of the file. These are the inputs to pyEIT image reconstruction.

    Format sources:
    - Reverse-engineered from PulmoVista 500 `.eit` output
    - Cross-referenced with EIDORS ``read_draeger_header`` /
      ``read_draeger_file`` (GPL, understanding only - no code copied)
    """

    def validate(self, path: Path) -> bool:
        """Return True if file contains a recognised .eit vendor magic string."""
        try:
            detect_vendor_from_eit_header(path)
            return True
        except (ValueError, OSError):
            return False

    def parse(self, path: Path) -> RawImpedanceData:
        """Parse `.eit` file into RawImpedanceData.

        Args:
            path: Path to the ``.eit`` file.

        Returns:
            ``RawImpedanceData`` with transimpedance measurements,
            sampling frequency, and header metadata.

        Raises:
            ValueError: If the file is not a valid Dräger .eit file.
        """
        path = Path(path)

        # ── 1. Read preamble to find sep_offset ───────────────────────────────
        preamble_size = PREAMBLE_N_FIELDS * PREAMBLE_DTYPE.itemsize

        if path.stat().st_size < preamble_size:
            raise ValueError(f"'{path}' is too small to be a valid .eit file.")

        preamble = np.memmap(
            path, dtype=PREAMBLE_DTYPE, mode="r", shape=(PREAMBLE_N_FIELDS,)
        )
        _, sep_offset, _ = int(preamble[0]), int(preamble[1]), int(preamble[2])

        with path.open("rb") as f:
            raw_header = f.read(sep_offset + len(SEPARATOR))

        # ── 2. Resolve spec by frame size divisibility ────────────────────────
        binary_start = sep_offset + len(SEPARATOR)
        data_size = path.stat().st_size - binary_start
        candidates = [
            s for s in get_eit_specs("draeger") if data_size % s.frame_size_bytes == 0
        ]
        if not candidates:
            known = [s.frame_size_bytes for s in get_eit_specs("draeger")]
            raise ValueError(
                f"No registered Dräger frame size divides data section "
                f"({data_size} bytes) of '{path}'. Known: {known}."
            )
        if len(candidates) > 1:
            names = [s.name for s in candidates]
            raise ValueError(
                f"Ambiguous Dräger .eit format: multiple specs match "
                f"data_size={data_size}. Candidates: {names}."
            )
        spec = candidates[0]

        # ── 3. Parse header metadata ──────────────────────────────────────────
        metadata, binary_start = parse_eit_header(raw_header)
        metadata["detected_spec"] = spec.name

        # ── 4. Memory-map binary frame data ──────────────────────────────────
        n_frames = data_size // spec.frame_size_bytes
        frames = np.memmap(
            path,
            dtype=FRAME_EIT_DTYPE,
            mode="r",
            offset=binary_start,
            shape=(n_frames,),
        )

        # ── 5. Extract fields and compute calibrated transimpedance ──────────
        # Calibration with the empirical constants of ``eit_utils`` (values from
        # EIDORS, A. Adler 2016; not present in the .eit header).
        trans_A = np.array(frames["trans_A"])  # (n_frames, 208) ADC counts
        trans_B = np.array(frames["trans_B"])  # (n_frames, 208) ADC counts
        inj_curr = np.array(frames["injection_current"])  # (n_frames, 16)  ADC counts
        voltage_A = np.array(frames["voltage_A"])  # (n_frames, 16)  ADC counts
        voltage_B = np.array(frames["voltage_B"])  # (n_frames, 16)  ADC counts

        transimpedance = (
            PRIMARY_TRANSIMPEDANCE_SCALE * trans_A
            - REFERENCE_TRANSIMPEDANCE_SCALE * trans_B
        )  # (n_frames, 208) calibrated transimpedance
        I_real = (
            inj_curr / CURRENT_COUNTS_PER_AMPERE
        )  # (n_frames, 16)  actual injected current [A]
        V_diff = (
            voltage_A - voltage_B
        ) / VOLTAGE_COUNTS_PER_VOLT  # (n_frames, 16)  differential voltage [V]

        aux_signals: dict[str, np.ndarray] = {
            "timestamp": np.array(frames["timestamp"]),  # fraction of day
            "trans_A": trans_A,  # raw, for audit
            "trans_B": trans_B,  # raw, for audit
            "injection_current": inj_curr,  # raw ADC counts
            "I_real": I_real,  # [A] actual injected current
            "voltage_A": voltage_A,  # raw ADC counts
            "voltage_B": voltage_B,  # raw ADC counts
            "V_diff": V_diff,  # [V] differential voltage
            "frame_counter": np.array(frames["frame_counter"]),
            "medibus": np.array(frames["medibus"]),  # (n_frames, 67)
        }

        # ── 6. Assemble result ────────────────────────────────────────────────
        if "fs" not in metadata:
            warnings.warn(
                f"'Framerate [Hz]' not found in header of '{path}'. "
                "Defaulting to fs=50.0 Hz.",
                UserWarning,
                stacklevel=2,
            )
        fs: float = metadata.get("fs", 50.0)
        metadata["n_frames"] = n_frames
        metadata["frame_format"] = spec.name
        metadata["n_electrodes"] = spec.n_electrodes
        metadata["n_measurements"] = spec.n_measurements

        return RawImpedanceData(
            measurements=transimpedance,
            aux_signals=aux_signals,
            fs=fs,
            filename=str(path),
            file_format="eit",
            vendor="draeger",
            metadata=metadata,
        )
