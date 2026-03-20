"""Tests for FRAME_BASE_DTYPE and FRAME_EXT_DTYPE (Task 1.2.1)."""

import numpy as np

from fasteit.dtypes import (
    FRAME_BASE_DTYPE,
    FRAME_EXT_DTYPE,
    MEDIBUS_EXT_FIELDS,
    MEDIBUS_EXT_INDEX,
    MEDIBUS_FIELDS,
    MEDIBUS_INDEX,
)

# ── itemsize ──────────────────────────────────────────────────────────────────


def test_frame_base_itemsize():
    assert FRAME_BASE_DTYPE.itemsize == 4358


def test_frame_ext_itemsize():
    assert FRAME_EXT_DTYPE.itemsize == 4382


# ── field types and shapes ────────────────────────────────────────────────────


def test_ts_is_float64():
    assert FRAME_BASE_DTYPE["ts"].itemsize == 8
    assert FRAME_BASE_DTYPE["ts"].base == np.dtype("<f8")


def test_pixels_shape():
    assert FRAME_BASE_DTYPE["pixels"].shape == (32, 32)
    assert FRAME_EXT_DTYPE["pixels"].shape == (32, 32)


def test_pixels_dtype():
    assert FRAME_BASE_DTYPE["pixels"].base == np.dtype("<f4")


def test_min_max_flag_is_int32():
    assert FRAME_BASE_DTYPE["min_max_flag"].base == np.dtype("<i4")


def test_event_marker_is_int32():
    assert FRAME_BASE_DTYPE["event_marker"].base == np.dtype("<i4")


def test_event_text_length():
    assert FRAME_BASE_DTYPE["event_text"].itemsize == 30


def test_medibus_base_shape():
    assert FRAME_BASE_DTYPE["medibus_data"].shape == (52,)


def test_medibus_ext_shape():
    assert FRAME_EXT_DTYPE["medibus_data"].shape == (58,)


def test_medibus_dtype():
    assert FRAME_BASE_DTYPE["medibus_data"].base == np.dtype("<f4")


# ── Medibus field lists ───────────────────────────────────────────────────────


def test_medibus_fields_count():
    assert len(MEDIBUS_FIELDS) == 52


def test_medibus_ext_fields_count():
    assert len(MEDIBUS_EXT_FIELDS) == 58


def test_medibus_fields_structure():
    for name, unit, is_continuous in MEDIBUS_FIELDS:
        assert isinstance(name, str) and name
        assert isinstance(unit, str)
        assert isinstance(is_continuous, bool)


def test_medibus_index_lookup():
    assert MEDIBUS_INDEX["airway_pressure"] == 0
    assert MEDIBUS_INDEX["flow"] == 1
    assert MEDIBUS_INDEX["volume"] == 2
    assert MEDIBUS_INDEX["peep"] == 14
    assert (
        MEDIBUS_INDEX["time_at_low_pressure"] == 51
    )  # Tlow — last field in BASE format
    assert (
        MEDIBUS_EXT_INDEX["high_pressure"] == 51
    )  # PHigh BiLevel — first EXT-specific field
    assert MEDIBUS_EXT_INDEX["low_pressure"] == 52  # Plow BiLevel
    assert MEDIBUS_EXT_INDEX["time_at_low_pressure"] == 53  # Tlow shifts 51→53 in EXT


def test_medibus_ext_index_includes_base():
    assert MEDIBUS_EXT_INDEX["airway_pressure"] == 0
    assert MEDIBUS_EXT_INDEX["peep"] == 14


def test_medibus_ext_pod_fields_present():
    assert "airway_pressure_pod" in MEDIBUS_EXT_INDEX
    assert "esophageal_pressure_pod" in MEDIBUS_EXT_INDEX
    assert "transpulmonary_pressure_pod" in MEDIBUS_EXT_INDEX
    assert "gastric_pressure_pod" in MEDIBUS_EXT_INDEX


def test_continuous_fields_at_start():
    """First 6 Medibus fields must be continuous waveforms."""
    for i in range(6):
        name, unit, is_continuous = MEDIBUS_FIELDS[i]
        assert is_continuous, f"Field {i} ({name}) expected continuous=True"


def test_non_continuous_fields_after_index_5():
    """Fields 6-51 must all be non-continuous (breath-averaged)."""
    for i in range(6, 52):
        name, unit, is_continuous = MEDIBUS_FIELDS[i]
        assert not is_continuous, f"Field {i} ({name}) expected continuous=False"


# ── round-trip with frombuffer ─────────────────────────────────────────────────


def test_frombuffer_base_dtype():
    """A zero-filled buffer of 4358 bytes must parse as one base frame."""
    buf = bytes(4358)
    arr = np.frombuffer(buf, dtype=FRAME_BASE_DTYPE)
    assert arr.shape == (1,)
    assert arr["pixels"].shape == (1, 32, 32)
    assert arr["medibus_data"].shape == (1, 52)


def test_frombuffer_ext_dtype():
    """A zero-filled buffer of 4382 bytes must parse as one extended frame."""
    buf = bytes(4382)
    arr = np.frombuffer(buf, dtype=FRAME_EXT_DTYPE)
    assert arr.shape == (1,)
    assert arr["medibus_data"].shape == (1, 58)


def test_frame_base_dtype_consistent_with_ext():
    """Base and extended dtypes share the same first 7 fields."""
    shared = [
        "ts",
        "dummy",
        "pixels",
        "min_max_flag",
        "event_marker",
        "event_text",
        "timing_error",
    ]
    for field in shared:
        assert FRAME_BASE_DTYPE[field] == FRAME_EXT_DTYPE[field]
