"""Timpel tabular schema constants and dtype templates.

This module captures the column layout inferred from the reference loader
provided in `.claude/timpel.py` (eitprocessing style):

- 1030 columns per frame
- cols 0..1023: 32x32 pixel impedance (row-major)
- col 1024: airway pressure
- col 1025: flow
- col 1026: volume
- col 1027: min marker flag
- col 1028: max marker flag
- col 1029: QRS marker flag

Notes:
- Source files are text/tabular (comma-separated), not fixed-width binary.
- Numeric parsing in upstream examples uses float, so values are represented
  here with float32 templates for downstream normalization.
"""

from __future__ import annotations

import numpy as np

TIMPEL_COLUMN_COUNT = 1030
TIMPEL_PIXEL_COUNT = 1024
TIMPEL_PIXEL_GRID_SHAPE = (32, 32)

TIMPEL_AIRWAY_PRESSURE_COL = 1024
TIMPEL_FLOW_COL = 1025
TIMPEL_VOLUME_COL = 1026
TIMPEL_MIN_FLAG_COL = 1027
TIMPEL_MAX_FLAG_COL = 1028
TIMPEL_QRS_FLAG_COL = 1029

TIMPEL_NUMERIC_DTYPE = np.float32
TIMPEL_NAN_SENTINEL = -1000.0
TIMPEL_DEFAULT_SAMPLE_FREQUENCY = 50.0

# Structured representation for one decoded frame row.
TIMPEL_FRAME_DTYPE = np.dtype(
    [
        ("pixels", "<f4", (TIMPEL_PIXEL_COUNT,)),
        ("airway_pressure", "<f4"),
        ("flow", "<f4"),
        ("volume", "<f4"),
        ("min_flag", "<f4"),
        ("max_flag", "<f4"),
        ("qrs_flag", "<f4"),
    ]
)
