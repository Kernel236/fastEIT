"""Tests for FRAME_EIT_DTYPE layout and field offsets."""

import numpy as np

from fasteit.parsers.draeger.eit.eit_dtypes import (
    FRAME_EIT_DTYPE,
    PREAMBLE_DTYPE,
    PREAMBLE_N_FIELDS,
)


# ── itemsize ──────────────────────────────────────────────────────────────────


def test_frame_eit_itemsize():
    assert FRAME_EIT_DTYPE.itemsize == 5495


def test_preamble_dtype_is_int32_le():
    assert PREAMBLE_DTYPE == np.dtype("<i4")


def test_preamble_n_fields():
    assert PREAMBLE_N_FIELDS == 3


# ── field types and shapes ────────────────────────────────────────────────────


def test_timestamp_is_float64():
    assert FRAME_EIT_DTYPE["timestamp"].base == np.dtype("<f8")


def test_trans_a_shape():
    assert FRAME_EIT_DTYPE["trans_A"].shape == (208,)
    assert FRAME_EIT_DTYPE["trans_A"].base == np.dtype("<f8")


def test_trans_b_shape():
    assert FRAME_EIT_DTYPE["trans_B"].shape == (208,)
    assert FRAME_EIT_DTYPE["trans_B"].base == np.dtype("<f8")


def test_injection_current_shape():
    assert FRAME_EIT_DTYPE["injection_current"].shape == (16,)


def test_voltage_a_shape():
    assert FRAME_EIT_DTYPE["voltage_A"].shape == (16,)


def test_voltage_b_shape():
    assert FRAME_EIT_DTYPE["voltage_B"].shape == (16,)


def test_medibus_shape():
    assert FRAME_EIT_DTYPE["medibus"].shape == (67,)
    assert FRAME_EIT_DTYPE["medibus"].base == np.dtype("<f4")


def test_frame_counter_is_uint16():
    assert FRAME_EIT_DTYPE["frame_counter"].base == np.dtype("<u2")


def test_event_text_length():
    assert FRAME_EIT_DTYPE["event_text"].itemsize == 30


# ── field offsets (documented in eit_dtypes.py module docstring) ──────────────


def test_trans_a_offset():
    """trans_A starts at byte 16: timestamp(8) + unknown_f8(8)."""
    assert FRAME_EIT_DTYPE.fields["trans_A"][1] == 16


def test_trans_b_offset():
    """trans_B starts at byte 2592 per reverse-engineering."""
    assert FRAME_EIT_DTYPE.fields["trans_B"][1] == 2592


# ── round-trip with frombuffer ────────────────────────────────────────────────


def test_frombuffer_single_frame():
    buf = bytes(5495)
    arr = np.frombuffer(buf, dtype=FRAME_EIT_DTYPE)
    assert arr.shape == (1,)
    assert arr["trans_A"].shape == (1, 208)
    assert arr["trans_B"].shape == (1, 208)
    assert arr["medibus"].shape == (1, 67)


def test_frombuffer_multi_frame():
    n = 5
    buf = bytes(5495 * n)
    arr = np.frombuffer(buf, dtype=FRAME_EIT_DTYPE)
    assert arr.shape == (n,)
