"""Tests for Timpel schema constants and TIMPEL_FRAME_DTYPE."""

import numpy as np

from fasteit.parsers.timpel.timpel_dtypes import (
    TIMPEL_AUX_FIELDS,
    TIMPEL_COLUMN_COUNT,
    TIMPEL_DEFAULT_SAMPLE_FREQUENCY,
    TIMPEL_FRAME_DTYPE,
    TIMPEL_NAN_SENTINEL,
    TIMPEL_PIXEL_COUNT,
    TIMPEL_PIXEL_GRID_SHAPE,
)


def test_column_count():
    assert TIMPEL_COLUMN_COUNT == 1030


def test_pixel_count():
    assert TIMPEL_PIXEL_COUNT == 1024


def test_pixel_grid_shape():
    assert TIMPEL_PIXEL_GRID_SHAPE == (32, 32)
    assert TIMPEL_PIXEL_GRID_SHAPE[0] * TIMPEL_PIXEL_GRID_SHAPE[1] == TIMPEL_PIXEL_COUNT


def test_default_fs():
    assert TIMPEL_DEFAULT_SAMPLE_FREQUENCY == 50.0


def test_nan_sentinel():
    assert TIMPEL_NAN_SENTINEL == -1000.0


def test_aux_fields_count():
    # airway_pressure, flow, volume, min_flag, max_flag, qrs_flag
    assert len(TIMPEL_AUX_FIELDS) == 6


def test_aux_fields_names():
    assert "airway_pressure" in TIMPEL_AUX_FIELDS
    assert "flow" in TIMPEL_AUX_FIELDS
    assert "volume" in TIMPEL_AUX_FIELDS
    assert "min_flag" in TIMPEL_AUX_FIELDS
    assert "max_flag" in TIMPEL_AUX_FIELDS
    assert "qrs_flag" in TIMPEL_AUX_FIELDS


def test_frame_dtype_has_ts_field():
    assert "ts" in TIMPEL_FRAME_DTYPE.names
    assert TIMPEL_FRAME_DTYPE["ts"] == np.dtype("<f8")


def test_frame_dtype_has_pixels_field():
    assert "pixels" in TIMPEL_FRAME_DTYPE.names
    assert TIMPEL_FRAME_DTYPE["pixels"].shape == (32, 32)
    assert TIMPEL_FRAME_DTYPE["pixels"].base == np.dtype("<f4")


def test_frame_dtype_no_aux_fields():
    """Aux signals go in aux_signals dict, not in the frame dtype."""
    for name in TIMPEL_AUX_FIELDS:
        assert name not in TIMPEL_FRAME_DTYPE.names


def test_frame_dtype_instantiation():
    """Verify TIMPEL_FRAME_DTYPE can create a valid structured array."""
    frames = np.zeros(5, dtype=TIMPEL_FRAME_DTYPE)
    assert frames.shape == (5,)
    assert frames["pixels"].shape == (5, 32, 32)
    assert frames["ts"].dtype == np.float64
