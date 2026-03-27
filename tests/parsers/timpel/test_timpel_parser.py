"""Tests for TimpelTabularParser — validate() and parse()."""

from pathlib import Path

import numpy as np
import pytest

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.errors import InvalidSliceError
from fasteit.parsers.timpel import TimpelTabularParser
from fasteit.parsers.timpel.timpel_dtypes import (
    TIMPEL_AUX_FIELDS,
    TIMPEL_COLUMN_COUNT,
    TIMPEL_DEFAULT_SAMPLE_FREQUENCY,
    TIMPEL_NAN_SENTINEL,
)

# ── Synthetic file helpers ────────────────────────────────────────────────────


def _make_timpel_csv(
    tmp_path: Path,
    n_frames: int = 5,
    sentinel_row: int | None = None,
) -> Path:
    """Write a minimal valid Timpel CSV with 1030 columns per row."""
    rng = np.random.default_rng(42)
    data = rng.uniform(0.1, 1.0, size=(n_frames, TIMPEL_COLUMN_COUNT)).astype(
        np.float64
    )
    data[:, 1024] = 10.0  # airway_pressure
    data[:, 1025] = 0.5  # flow
    data[:, 1026] = 0.3  # volume
    data[:, 1027] = 0.0  # min_flag (off)
    data[:, 1028] = 0.0  # max_flag (off)
    data[:, 1029] = 0.0  # qrs_flag (off)
    data[0, 1027] = 1.0  # plant one min marker on first frame
    data[2, 1028] = 1.0  # plant one max marker on third frame

    if sentinel_row is not None:
        data[sentinel_row, :1024] = TIMPEL_NAN_SENTINEL  # pixel sentinel
        data[sentinel_row, 1024] = TIMPEL_NAN_SENTINEL  # pressure sentinel

    p = tmp_path / "recording.csv"
    np.savetxt(str(p), data, delimiter=",", fmt="%.6f")
    return p


# ── validate() ───────────────────────────────────────────────────────────────


def test_validate_true_for_valid_file(tmp_path):
    p = _make_timpel_csv(tmp_path)
    assert TimpelTabularParser().validate(p) is True


def test_validate_false_missing_file(tmp_path):
    assert TimpelTabularParser().validate(tmp_path / "ghost.csv") is False


def test_validate_false_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert TimpelTabularParser().validate(p) is False


def test_validate_false_wrong_extension(tmp_path):
    p = _make_timpel_csv(tmp_path)
    p2 = p.rename(tmp_path / "recording.bin")
    assert TimpelTabularParser().validate(p2) is False


def test_validate_true_for_txt_extension(tmp_path):
    """validate() must accept .txt as well as .csv."""
    p = _make_timpel_csv(tmp_path)
    p_txt = p.rename(tmp_path / "recording.txt")
    assert TimpelTabularParser().validate(p_txt) is True


def test_validate_false_for_short_csv(tmp_path):
    """A CSV with fewer than 1030 columns should not be detected as Timpel."""
    p = tmp_path / "short.csv"
    # Write 10 columns — not a Timpel file
    np.savetxt(str(p), np.zeros((5, 10)), delimiter=",")
    assert TimpelTabularParser().validate(p) is False


# ── parse() — basic contract ─────────────────────────────────────────────────


def test_parse_returns_reconstructed_frame_data(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    assert isinstance(data, ReconstructedFrameData)


def test_parse_frame_count(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=7)
    data = TimpelTabularParser().parse(p)
    assert data.n_frames == 7


def test_parse_file_format_is_csv(tmp_path):
    p = _make_timpel_csv(tmp_path)
    data = TimpelTabularParser().parse(p)
    assert data.file_format == "csv"


def test_parse_file_format_from_extension(tmp_path):
    """file_format should reflect the actual file extension, not a hardcoded string."""
    p = _make_timpel_csv(tmp_path)
    p_txt = p.rename(tmp_path / "recording.txt")
    data = TimpelTabularParser().parse(p_txt)
    assert data.file_format == "txt"


def test_parse_fs_is_50hz(tmp_path):
    p = _make_timpel_csv(tmp_path)
    data = TimpelTabularParser().parse(p)
    assert data.fs == TIMPEL_DEFAULT_SAMPLE_FREQUENCY


# ── parse() — frames array ───────────────────────────────────────────────────


def test_parse_pixels_shape(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    assert data.pixels.shape == (5, 32, 32)


def test_parse_pixels_dtype_float32(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=3)
    data = TimpelTabularParser().parse(p)
    assert data.pixels.dtype == np.float32


def test_parse_timestamps_start_at_zero(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    assert data.timestamps[0] == pytest.approx(0.0)


def test_parse_timestamps_spacing_matches_fs(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    dt = np.diff(data.timestamps)
    assert np.allclose(dt, 1.0 / TIMPEL_DEFAULT_SAMPLE_FREQUENCY)


# ── parse() — aux_signals ─────────────────────────────────────────────────────


def test_parse_aux_signals_present(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    assert data.aux_signals is not None


def test_parse_aux_signals_all_fields(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    for field in TIMPEL_AUX_FIELDS:
        assert field in data.aux_signals, f"Missing aux field: {field}"


def test_parse_aux_signals_shape(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    for field, arr in data.aux_signals.items():
        assert arr.shape == (5,), f"{field}: expected (5,), got {arr.shape}"


def test_parse_aux_pressure_value(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    assert np.allclose(data.aux_signals["airway_pressure"], 10.0)


def test_parse_min_flag_marker(tmp_path):
    """First frame has min_flag=1 planted by fixture."""
    p = _make_timpel_csv(tmp_path, n_frames=5)
    data = TimpelTabularParser().parse(p)
    assert data.aux_signals["min_flag"][0] == pytest.approx(1.0)
    assert data.aux_signals["min_flag"][1] == pytest.approx(0.0)


# ── parse() — NaN sentinel replacement ───────────────────────────────────────


def test_parse_pixel_sentinel_replaced_with_nan(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5, sentinel_row=1)
    data = TimpelTabularParser().parse(p)
    assert np.all(np.isnan(data.pixels[1]))
    assert not np.any(np.isnan(data.pixels[0]))


def test_parse_pressure_sentinel_replaced_with_nan(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5, sentinel_row=1)
    data = TimpelTabularParser().parse(p)
    assert np.isnan(data.aux_signals["airway_pressure"][1])
    assert not np.isnan(data.aux_signals["airway_pressure"][0])


# ── parse() — slicing ─────────────────────────────────────────────────────────


def test_parse_first_frame_offset(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=10)
    data = TimpelTabularParser().parse(p, first_frame=5)
    assert data.n_frames == 5
    # Synthetic timestamps offset: frame 5 at fs=50 → t=0.1 s
    assert data.timestamps[0] == pytest.approx(5 / 50.0)


def test_parse_max_frames(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=10)
    data = TimpelTabularParser().parse(p, max_frames=3)
    assert data.n_frames == 3


def test_parse_first_frame_and_max_frames(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=10)
    data = TimpelTabularParser().parse(p, first_frame=2, max_frames=4)
    assert data.n_frames == 4
    assert data.timestamps[0] == pytest.approx(2 / 50.0)


def test_parse_first_frame_out_of_bounds_raises(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    with pytest.raises(InvalidSliceError, match="beyond the end"):
        TimpelTabularParser().parse(p, first_frame=100)


def test_parse_negative_first_frame_raises(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    with pytest.raises(InvalidSliceError):
        TimpelTabularParser().parse(p, first_frame=-1)


def test_parse_zero_max_frames_raises(tmp_path):
    p = _make_timpel_csv(tmp_path, n_frames=5)
    with pytest.raises(InvalidSliceError):
        TimpelTabularParser().parse(p, max_frames=0)


# ── parse() — schema errors ───────────────────────────────────────────────────


def test_parse_wrong_column_count_raises(tmp_path):
    p = tmp_path / "bad.csv"
    np.savetxt(str(p), np.zeros((5, 100)), delimiter=",")
    with pytest.raises(ValueError, match="1030 columns"):
        TimpelTabularParser().parse(p)


# ── parse() — pixel round-trip precision ─────────────────────────────────────


def test_parse_pixel_value_survives_roundtrip(tmp_path):
    """Known pixel value must survive savetxt→loadtxt→float32 within 1e-5."""
    rng = np.random.default_rng(0)
    data = rng.uniform(0.1, 1.0, size=(3, TIMPEL_COLUMN_COUNT)).astype(np.float64)
    data[:, 1024:] = 0.0  # zero aux columns
    data[0, 0] = 0.123456  # known value at pixel (0, 0)
    p = tmp_path / "roundtrip.csv"
    np.savetxt(str(p), data, delimiter=",", fmt="%.6f")

    result = TimpelTabularParser().parse(p)

    assert result.pixels[0, 0, 0] == pytest.approx(0.123456, abs=1e-5)
