"""Timpel file schema and frame dtype."""

import numpy as np

from fasteit.parsers.timpel.timpel_dtypes import (
    TIMPEL_AUX_FIELDS,
    TIMPEL_AUX_UNITS,
    TIMPEL_COLUMN_COUNT,
    TIMPEL_DEFAULT_SAMPLE_FREQUENCY,
    TIMPEL_FRAME_DTYPE,
    TIMPEL_NAN_SENTINEL,
    TIMPEL_PIXEL_COUNT,
    TIMPEL_PIXEL_GRID_SHAPE,
)


def test_file_schema():
    assert TIMPEL_COLUMN_COUNT == 1030  # 1024 pixels + 6 auxiliary columns
    assert TIMPEL_PIXEL_COUNT == 1024 == np.prod(TIMPEL_PIXEL_GRID_SHAPE)
    assert TIMPEL_PIXEL_GRID_SHAPE == (32, 32)
    assert TIMPEL_DEFAULT_SAMPLE_FREQUENCY == 50.0
    assert TIMPEL_NAN_SENTINEL == -1000.0
    assert TIMPEL_AUX_FIELDS == (
        "airway_pressure", "flow", "volume", "min_flag", "max_flag", "qrs_flag",
    )  # fmt: skip
    assert set(TIMPEL_AUX_UNITS) == set(TIMPEL_AUX_FIELDS)


def test_frame_dtype():
    assert TIMPEL_FRAME_DTYPE["ts"] == np.dtype("<f8")
    assert TIMPEL_FRAME_DTYPE["pixels"].base == np.dtype("<f4")
    assert TIMPEL_FRAME_DTYPE["pixels"].shape == (32, 32)
    # Auxiliary signals go in ``aux_signals``, not in the frames.
    assert not set(TIMPEL_AUX_FIELDS) & set(TIMPEL_FRAME_DTYPE.names)
    frames = np.zeros(5, dtype=TIMPEL_FRAME_DTYPE)
    assert frames["pixels"].shape == (5, 32, 32)
