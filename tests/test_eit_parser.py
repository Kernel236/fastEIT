"""Tests for DragerEitParser using synthetic and real .eit files."""

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from fasteit.models.raw_impedance_data import RawImpedanceData
from fasteit.parsers.draeger.eit.eit_dtypes import FRAME_EIT_DTYPE
from fasteit.parsers.draeger.eit.eit_parser import DragerEitParser
from fasteit.parsers.draeger.eit.eit_utils import FT_A, FT_B, SEPARATOR

# ── real test files (skipped on CI if absent) ─────────────────────────────────

_EIT_REAL = Path(__file__).parent.parent / "src/fasteit/test_files/patient01.eit"
_real = pytest.mark.skipif(not _EIT_REAL.exists(), reason="real .eit file not available")


# ── synthetic .eit file builder ───────────────────────────────────────────────


def _write_eit(
    tmp_path: Path,
    n_frames: int = 5,
    fs: float = 50.0,
    trans_A_val: float = 1.0,
    trans_B_val: float = 0.5,
    magic: str = "Draeger EIT-Software",
) -> Path:
    """Write a minimal synthetic Dräger .eit file with known frame values."""
    header_lines = [
        magic,
        f"Framerate [Hz]: {fs}",
        "Date: 01.01.2024",
        "Time: 12:00:00",
        "Gain: 65",
        "Frequency [kHz]: 101.5",
    ]
    header_text = "\r\n".join(header_lines) + "\r\n"
    header_bytes = header_text.encode("latin-1")

    preamble_size = 3 * 4
    sep_offset = preamble_size + len(header_bytes)
    preamble = struct.pack("<iii", 51, sep_offset, 0)

    frames = np.zeros(n_frames, dtype=FRAME_EIT_DTYPE)
    dt_day = 1.0 / (fs * 86400.0)
    frames["timestamp"] = np.arange(n_frames) * dt_day
    frames["trans_A"] = trans_A_val
    frames["trans_B"] = trans_B_val
    frames["frame_counter"] = np.arange(n_frames, dtype=np.uint16)

    path = tmp_path / "synthetic.eit"
    with path.open("wb") as f:
        f.write(preamble)
        f.write(header_bytes)
        f.write(SEPARATOR)
        frames.tofile(f)
    return path


# ── validate() ────────────────────────────────────────────────────────────────


def test_validate_true_for_synthetic(tmp_path):
    p = _write_eit(tmp_path)
    assert DragerEitParser().validate(p)


def test_validate_false_for_missing_file(tmp_path):
    assert not DragerEitParser().validate(tmp_path / "ghost.eit")


def test_validate_false_for_no_magic_string(tmp_path):
    p = _write_eit(tmp_path, magic="SomeOtherVendor EIT")
    assert not DragerEitParser().validate(p)


def test_validate_false_for_too_small_file(tmp_path):
    p = tmp_path / "tiny.eit"
    p.write_bytes(b"\x00" * 4)
    assert not DragerEitParser().validate(p)


# ── parse() — return type and structure ──────────────────────────────────────


def test_parse_returns_raw_impedance_data(tmp_path):
    p = _write_eit(tmp_path)
    data = DragerEitParser().parse(p)
    assert isinstance(data, RawImpedanceData)


def test_parse_measurements_shape(tmp_path):
    p = _write_eit(tmp_path, n_frames=10)
    data = DragerEitParser().parse(p)
    assert data.measurements.shape == (10, 208)


def test_parse_fs_from_header(tmp_path):
    p = _write_eit(tmp_path, fs=25.0)
    data = DragerEitParser().parse(p)
    assert data.fs == pytest.approx(25.0)


def test_parse_vendor_is_draeger(tmp_path):
    p = _write_eit(tmp_path)
    data = DragerEitParser().parse(p)
    assert data.vendor == "draeger"


def test_parse_file_format_is_eit(tmp_path):
    p = _write_eit(tmp_path)
    data = DragerEitParser().parse(p)
    assert data.file_format == "eit"


# ── parse() — metadata ────────────────────────────────────────────────────────


def test_parse_metadata_n_frames(tmp_path):
    p = _write_eit(tmp_path, n_frames=7)
    data = DragerEitParser().parse(p)
    assert data.metadata["n_frames"] == 7


def test_parse_metadata_n_electrodes(tmp_path):
    p = _write_eit(tmp_path)
    data = DragerEitParser().parse(p)
    assert data.metadata["n_electrodes"] == 16


def test_parse_metadata_n_measurements(tmp_path):
    p = _write_eit(tmp_path)
    data = DragerEitParser().parse(p)
    assert data.metadata["n_measurements"] == 208


def test_parse_metadata_date(tmp_path):
    p = _write_eit(tmp_path)
    data = DragerEitParser().parse(p)
    assert data.metadata["date"] == "01.01.2024"


# ── parse() — calibration vv = FT_A*trans_A - FT_B*trans_B ──────────────────


def test_parse_measurements_calibration(tmp_path):
    """vv must equal FT_A*trans_A - FT_B*trans_B."""
    p = _write_eit(tmp_path, trans_A_val=2.0, trans_B_val=1.0)
    data = DragerEitParser().parse(p)
    expected = FT_A * 2.0 - FT_B * 1.0
    assert np.allclose(data.measurements, expected)


def test_parse_measurements_no_nan(tmp_path):
    p = _write_eit(tmp_path)
    assert not np.isnan(data := DragerEitParser().parse(p).measurements).any()


def test_parse_measurements_not_all_zero(tmp_path):
    p = _write_eit(tmp_path, trans_A_val=1.0, trans_B_val=0.0)
    data = DragerEitParser().parse(p)
    assert not np.all(data.measurements == 0)


# ── parse() — aux_signals ─────────────────────────────────────────────────────


def test_parse_aux_signals_keys(tmp_path):
    p = _write_eit(tmp_path)
    data = DragerEitParser().parse(p)
    expected_keys = {
        "timestamp", "trans_A", "trans_B",
        "injection_current", "I_real",
        "voltage_A", "voltage_B", "V_diff",
        "frame_counter", "medibus",
    }
    assert expected_keys.issubset(set(data.aux_signals.keys()))


def test_parse_aux_signals_timestamp_monotonic(tmp_path):
    p = _write_eit(tmp_path, n_frames=10)
    data = DragerEitParser().parse(p)
    ts = data.aux_signals["timestamp"]
    assert np.all(np.diff(ts) >= 0)


def test_parse_aux_signals_frame_counter(tmp_path):
    p = _write_eit(tmp_path, n_frames=5)
    data = DragerEitParser().parse(p)
    fc = data.aux_signals["frame_counter"]
    assert fc.shape == (5,)
    np.testing.assert_array_equal(fc, np.arange(5, dtype=np.uint16))


def test_parse_aux_signals_trans_a_raw(tmp_path):
    """trans_A in aux_signals must be the raw values, not vv."""
    p = _write_eit(tmp_path, trans_A_val=3.0)
    data = DragerEitParser().parse(p)
    assert np.allclose(data.aux_signals["trans_A"], 3.0)


def test_parse_aux_signals_i_real_shape(tmp_path):
    p = _write_eit(tmp_path, n_frames=5)
    data = DragerEitParser().parse(p)
    assert data.aux_signals["I_real"].shape == (5, 16)


def test_parse_aux_signals_medibus_shape(tmp_path):
    p = _write_eit(tmp_path, n_frames=5)
    data = DragerEitParser().parse(p)
    assert data.aux_signals["medibus"].shape == (5, 67)


# ── parse() — fs fallback warning ─────────────────────────────────────────────


def test_parse_warns_if_framerate_missing(tmp_path):
    """If Framerate [Hz] is absent from header, a UserWarning must be raised."""
    header_lines = ["Draeger EIT-Software", "Date: 01.01.2024"]
    header_text = "\r\n".join(header_lines) + "\r\n"
    header_bytes = header_text.encode("latin-1")
    sep_offset = 12 + len(header_bytes)
    preamble = struct.pack("<iii", 51, sep_offset, 0)

    frames = np.zeros(3, dtype=FRAME_EIT_DTYPE)
    frames["timestamp"] = np.arange(3) * (1.0 / (50.0 * 86400.0))

    p = tmp_path / "no_framerate.eit"
    with p.open("wb") as f:
        f.write(preamble)
        f.write(header_bytes)
        f.write(SEPARATOR)
        frames.tofile(f)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = DragerEitParser().parse(p)

    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert data.fs == pytest.approx(50.0)


# ── parse() — error cases ─────────────────────────────────────────────────────


def test_parse_raises_for_too_small_file(tmp_path):
    p = tmp_path / "tiny.eit"
    p.write_bytes(b"\x00" * 4)
    with pytest.raises(ValueError):
        DragerEitParser().parse(p)


def test_parse_raises_for_truncated_binary(tmp_path):
    """Binary section not divisible by frame size must raise ValueError."""
    p = _write_eit(tmp_path, n_frames=3)
    data = p.read_bytes()
    p.write_bytes(data[:-100])  # truncate last 100 bytes
    with pytest.raises(ValueError, match="frame size"):
        DragerEitParser().parse(p)


# ── real file tests (skipped if file absent) ──────────────────────────────────


@_real
def test_real_parse_shape():
    data = DragerEitParser().parse(_EIT_REAL)
    n, m = data.measurements.shape
    assert n > 0
    assert m == 208


@_real
def test_real_fs_is_50():
    data = DragerEitParser().parse(_EIT_REAL)
    assert data.fs == pytest.approx(50.0)


@_real
def test_real_measurements_no_nan():
    data = DragerEitParser().parse(_EIT_REAL)
    assert not np.isnan(data.measurements).any()


@_real
def test_real_timestamps_monotonic():
    data = DragerEitParser().parse(_EIT_REAL)
    ts = data.aux_signals["timestamp"]
    assert np.all(np.diff(ts) >= 0)


@_real
def test_real_n_frames_coherent_with_duration():
    """n_frames / fs must match timestamp duration within 1 second."""
    data = DragerEitParser().parse(_EIT_REAL)
    ts = data.aux_signals["timestamp"]
    duration_ts = (ts[-1] - ts[0]) * 86400.0  # fraction-of-day → seconds
    duration_frames = data.metadata["n_frames"] / data.fs
    assert abs(duration_ts - duration_frames) < 1.0
