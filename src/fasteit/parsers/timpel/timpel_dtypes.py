"""Timpel tabular file schema constants.

Column layout (1030 columns per frame, comma-separated CSV, no header):

  cols  0–1023  pixel impedance — 32×32 reconstructed image (row-major)
  col  1024     airway_pressure  (cmH₂O)
  col  1025     flow             (L/s)
  col  1026     volume           (L)
  col  1027     min_flag         binary: 1 = expiration trough detected
  col  1028     max_flag         binary: 1 = inspiration peak detected
  col  1029     qrs_flag         binary: 1 = QRS complex detected

Sentinel: −1000.0 → NaN (same convention as Dräger Medibus channels).

Sampling frequency: 50 Hz fixed (no timestamp column in Timpel files).
Synthetic timestamps are generated as seconds from start of recording,
consistent with eitprocessing (Apache-2.0):
https://github.com/EIT-ALIVE/eitprocessing

Format reference: cross-validated against eitprocessing timpel.py loader.
Somhorst P et al., "eitprocessing", JOSS 2026;11(117):8179
DOI: 10.21105/joss.08179
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# File-level constants
# ---------------------------------------------------------------------------

TIMPEL_COLUMN_COUNT: int = 1030
TIMPEL_PIXEL_COUNT: int = 1024
TIMPEL_PIXEL_GRID_SHAPE: tuple[int, int] = (32, 32)

# Auxiliary signal column indices (after the 1024 pixel columns)
TIMPEL_AIRWAY_PRESSURE_COL: int = 1024
TIMPEL_FLOW_COL: int = 1025
TIMPEL_VOLUME_COL: int = 1026
TIMPEL_MIN_FLAG_COL: int = 1027
TIMPEL_MAX_FLAG_COL: int = 1028
TIMPEL_QRS_FLAG_COL: int = 1029

TIMPEL_NAN_SENTINEL: float = -1000.0
TIMPEL_DEFAULT_SAMPLE_FREQUENCY: float = 50.0  # Hz — fixed by device firmware

# ---------------------------------------------------------------------------
# Ordered auxiliary signal field names (maps col offset → dict key)
# ---------------------------------------------------------------------------
# Index i corresponds to column TIMPEL_AIRWAY_PRESSURE_COL + i.

TIMPEL_AUX_FIELDS: tuple[str, ...] = (
    "airway_pressure",  # col 1024, cmH₂O
    "flow",  # col 1025, L/s
    "volume",  # col 1026, L
    "min_flag",  # col 1027, binary (expiration trough)
    "max_flag",  # col 1028, binary (inspiration peak)
    "qrs_flag",  # col 1029, binary (QRS complex)
)

# ---------------------------------------------------------------------------
# Structured dtype for ReconstructedFrameData.frames
# ---------------------------------------------------------------------------
# Compatible with DragerBinParser output: has "ts" and "pixels" fields.
# "ts" holds synthetic time in seconds from start of recording.
# (Dräger stores fraction-of-day wall-clock time; Timpel has no timestamps.)

TIMPEL_FRAME_DTYPE: np.dtype = np.dtype(
    [
        ("ts", "<f8"),  # synthetic time in seconds from start
        ("pixels", "<f4", (32, 32)),  # reconstructed 32×32 image
    ]
)
# Total in-memory size per frame: 8 + 4*1024 = 4104 bytes
# (source file stores 1030 float64 values per row — not binary)
