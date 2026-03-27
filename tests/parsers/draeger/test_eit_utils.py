"""Tests for parse_eit_header() and HEADER_FIELD_MAP."""

import struct

import pytest

from fasteit.parsers.draeger.eit.eit_utils import (
    FC_CURRENT,
    FT_A,
    FT_B,
    FV_VOLTAGE,
    HEADER_FIELD_MAP,
    SEPARATOR,
    parse_eit_header,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_raw_header(fields: dict[str, str] | None = None) -> bytes:
    """Build a minimal synthetic .eit header buffer.

    Preamble: format_version=51, sep_offset=<computed>, unknown=0
    ASCII header: key: value\\r\\n lines (latin-1)
    Separator: b'**\\r\\n\\r\\n\\r\\n'
    """
    lines = []
    if fields:
        for k, v in fields.items():
            lines.append(f"{k}: {v}")
    header_text = "\r\n".join(lines) + "\r\n" if lines else "\r\n"
    header_bytes = header_text.encode("latin-1")

    preamble_size = 3 * 4  # 3 × int32 = 12 bytes
    sep_offset = preamble_size + len(header_bytes)

    preamble = struct.pack("<iii", 51, sep_offset, 0)
    return preamble + header_bytes + SEPARATOR


# ── SEPARATOR constant ────────────────────────────────────────────────────────


def test_separator_value():
    assert SEPARATOR == b"**\r\n\r\n\r\n"


def test_separator_length():
    assert len(SEPARATOR) == 8


# ── calibration constants ─────────────────────────────────────────────────────


def test_ft_a_value():
    assert FT_A == pytest.approx(0.00098242)


def test_ft_b_value():
    assert FT_B == pytest.approx(0.00019607)


def test_fc_current_value():
    assert FC_CURRENT == pytest.approx(194326.3536)


def test_fv_voltage_value():
    assert FV_VOLTAGE == pytest.approx(0.11771)


# ── HEADER_FIELD_MAP keys ─────────────────────────────────────────────────────


def test_framerate_key_in_field_map():
    assert "Framerate [Hz]" in HEADER_FIELD_MAP


def test_frequency_key_in_field_map():
    assert "Frequency [kHz]" in HEADER_FIELD_MAP


def test_no_calibration_factor_key():
    """Calibration Factor is not in the header — hardcoded from EIDORS."""
    assert "Calibration Factor" not in HEADER_FIELD_MAP


# ── parse_eit_header() — valid input ─────────────────────────────────────────


def test_parse_returns_dict_and_int():
    raw = _make_raw_header({"Framerate [Hz]": "50.0"})
    meta, binary_start = parse_eit_header(raw)
    assert isinstance(meta, dict)
    assert isinstance(binary_start, int)


def test_parse_binary_start_after_separator():
    raw = _make_raw_header()
    _, binary_start = parse_eit_header(raw)
    assert binary_start == len(raw)


def test_parse_fs_from_framerate_field():
    raw = _make_raw_header({"Framerate [Hz]": "25.6"})
    meta, _ = parse_eit_header(raw)
    assert meta["fs"] == pytest.approx(25.6)


def test_parse_fs_is_float():
    raw = _make_raw_header({"Framerate [Hz]": "50"})
    meta, _ = parse_eit_header(raw)
    assert isinstance(meta["fs"], float)


def test_parse_date_and_time():
    raw = _make_raw_header({"Date": "01.01.2024", "Time": "14:30:00"})
    meta, _ = parse_eit_header(raw)
    assert meta["date"] == "01.01.2024"
    assert meta["time"] == "14:30:00"


def test_parse_time_with_colon_in_value():
    """Time field contains colons — partition(':') must use first colon only."""
    raw = _make_raw_header({"Time": "14:30:00.123"})
    meta, _ = parse_eit_header(raw)
    assert meta["time"] == "14:30:00.123"


def test_parse_gain_as_int():
    raw = _make_raw_header({"Gain": "65"})
    meta, _ = parse_eit_header(raw)
    assert meta["gain"] == 65
    assert isinstance(meta["gain"], int)


def test_parse_frequency_khz():
    raw = _make_raw_header({"Frequency [kHz]": "101.5"})
    meta, _ = parse_eit_header(raw)
    assert meta["frequency_khz"] == pytest.approx(101.5)


def test_parse_format_version_in_metadata():
    raw = _make_raw_header()
    meta, _ = parse_eit_header(raw)
    assert meta["format_version"] == 51


def test_parse_binary_start_in_metadata():
    raw = _make_raw_header()
    meta, binary_start = parse_eit_header(raw)
    assert meta["binary_start"] == binary_start


def test_parse_raw_fields_present():
    raw = _make_raw_header({"Date": "01.01.2024", "Gain": "65"})
    meta, _ = parse_eit_header(raw)
    assert "_raw_fields" in meta
    assert meta["_raw_fields"]["Date"] == "01.01.2024"
    assert meta["_raw_fields"]["Gain"] == "65"


def test_parse_unknown_fields_kept_in_raw_fields():
    raw = _make_raw_header({"SomeUnknownField": "value123"})
    meta, _ = parse_eit_header(raw)
    assert "SomeUnknownField" in meta["_raw_fields"]
    assert "SomeUnknownField" not in meta  # not promoted to structured key


def test_parse_conversion_failure_keeps_string():
    """If a numeric field has a non-numeric value, it stays as string."""
    raw = _make_raw_header({"Gain": "not_a_number"})
    meta, _ = parse_eit_header(raw)
    assert meta["gain"] == "not_a_number"


# ── parse_eit_header() — error cases ─────────────────────────────────────────


def test_parse_raises_if_buffer_too_small():
    with pytest.raises(ValueError, match="too small"):
        parse_eit_header(b"\x00" * 4)


def test_parse_raises_if_sep_offset_out_of_range():
    # sep_offset points beyond the buffer
    bad_preamble = struct.pack("<iii", 51, 9999, 0)
    with pytest.raises(ValueError, match="sep_offset"):
        parse_eit_header(bad_preamble + b"\x00" * 20)
