"""Tests for DragerBinParser using a synthetic .bin file."""

from pathlib import Path

import numpy as np
import pytest

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.draeger import DragerBinParser
from fasteit.parsers.draeger.bin.draeger_dtypes import (
    FRAME_BASE_DTYPE,
    FRAME_EXT_DTYPE,
    MEDIBUS_EXT_FIELDS,
)

# ── Synthetic file fixture ────────────────────────────────────────────────────


def _write_bin(tmp_path: Path, dtype, n_frames: int = 10) -> Path:
    """Write a synthetic Dräger .bin file with zeros and predictable timestamps."""
    frames = np.zeros(n_frames, dtype=dtype)
    # timestamps: 0.00, 0.02, 0.04 ... fraction of day @ 50 Hz
    dt_day = 1.0 / (50.0 * 86400.0)
    frames["ts"] = np.arange(n_frames) * dt_day
    frames["pixels"][:, 16, 16] = 1.0  # one lit pixel per frame

    path = tmp_path / f"synthetic_{dtype.itemsize}b.bin"
    frames.tofile(path)
    return path


# ── validate() ────────────────────────────────────────────────────────────────


def test_validate_true_for_base_frame(tmp_path):
    p = _write_bin(tmp_path, FRAME_BASE_DTYPE)
    assert DragerBinParser().validate(p)


def test_validate_true_for_ext_frame(tmp_path):
    p = _write_bin(tmp_path, FRAME_EXT_DTYPE)
    assert DragerBinParser().validate(p)


def test_validate_false_for_missing_file(tmp_path):
    assert not DragerBinParser().validate(tmp_path / "ghost.bin")


def test_validate_false_for_wrong_size(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"\x00" * 100)  # not divisible by any known frame size
    assert not DragerBinParser().validate(p)


# ── parse() — BASE format ─────────────────────────────────────────────────────


def test_parse_base_returns_reconstructed_frame_data(tmp_path):
    p = _write_bin(tmp_path, FRAME_BASE_DTYPE, n_frames=10)
    data = DragerBinParser().parse(p)
    assert isinstance(data, ReconstructedFrameData)


def test_parse_base_frame_count(tmp_path):
    p = _write_bin(tmp_path, FRAME_BASE_DTYPE, n_frames=10)
    data = DragerBinParser().parse(p)
    assert data.n_frames == 10


def test_parse_base_fs_estimated(tmp_path):
    p = _write_bin(tmp_path, FRAME_BASE_DTYPE, n_frames=100)
    data = DragerBinParser().parse(p)
    assert data.fs is not None
    assert abs(data.fs - 50.0) < 1.0


def test_parse_base_pixels_shape(tmp_path):
    p = _write_bin(tmp_path, FRAME_BASE_DTYPE, n_frames=5)
    data = DragerBinParser().parse(p)
    assert data.pixels.shape == (5, 32, 32)


def test_parse_base_has_aux_signals(tmp_path):
    """BASE format includes 52 Medibus channels in aux_signals."""
    from fasteit.parsers.draeger.bin.draeger_dtypes import MEDIBUS_BASE_FIELDS

    p = _write_bin(tmp_path, FRAME_BASE_DTYPE, n_frames=5)
    data = DragerBinParser().parse(p)
    assert data.aux_signals is not None
    assert len(data.aux_signals) == len(MEDIBUS_BASE_FIELDS)


# ── parse() — EXT format ─────────────────────────────────────────────────────


def test_parse_ext_has_aux_signals(tmp_path):
    p = _write_bin(tmp_path, FRAME_EXT_DTYPE, n_frames=5)
    data = DragerBinParser().parse(p)
    assert data.aux_signals is not None
    assert len(data.aux_signals) == len(MEDIBUS_EXT_FIELDS)


def test_parse_ext_aux_signals_shape(tmp_path):
    p = _write_bin(tmp_path, FRAME_EXT_DTYPE, n_frames=5)
    data = DragerBinParser().parse(p)
    for arr in data.aux_signals.values():
        assert arr.shape == (5,)


# ── first_frame / max_frames slicing ─────────────────────────────────────────


def test_parse_slice_first_frame(tmp_path):
    p = _write_bin(tmp_path, FRAME_BASE_DTYPE, n_frames=20)
    data = DragerBinParser().parse(p, first_frame=5)
    assert data.n_frames == 15


def test_parse_slice_max_frames(tmp_path):
    p = _write_bin(tmp_path, FRAME_BASE_DTYPE, n_frames=20)
    data = DragerBinParser().parse(p, max_frames=7)
    assert data.n_frames == 7


# ── float sentinel replacement ────────────────────────────────────────────────


def test_parse_float_sentinel_preserved_in_raw_data(tmp_path):
    """Parser returns raw memmap — sentinel values are NOT replaced.

    Sentinel replacement is deferred to the preprocessing layer so that
    the memmap is never copied to RAM during parsing (lazy loading).
    """
    frames = np.zeros(3, dtype=FRAME_BASE_DTYPE)
    dt_day = 1.0 / (50.0 * 86400.0)
    frames["ts"] = np.arange(3) * dt_day
    frames["pixels"][1, 0, 0] = -1000.0  # float sentinel value
    path = tmp_path / "sentinel.bin"
    frames.tofile(path)

    data = DragerBinParser().parse(path)
    assert data.pixels[1, 0, 0] == -1000.0  


# ── Round-trip value correctness (Tasks 1.5.3 / 1.5.4 / 1.5.5 / 1.5.10) ─────

_DT_DAY = 1.0 / (50.0 * 86400.0)


def test_n_frames_from_fixture(bin_3frames):
    """Task 1.5.3 — frame count matches what was written."""
    data = DragerBinParser().parse(bin_3frames)
    assert data.n_frames == 3


def test_pixel_values_per_frame(bin_3frames):
    """Task 1.5.4 — every pixel in frame i equals float(i+1)."""
    data = DragerBinParser().parse(bin_3frames)
    for i in range(3):
        assert data.pixels[i].mean() == pytest.approx(float(i + 1))


def test_pixel_specific_position(bin_3frames):
    """Task 1.5.4 — spot-check two specific pixel positions."""
    data = DragerBinParser().parse(bin_3frames)
    assert data.pixels[0, 0, 0] == pytest.approx(1.0)
    assert data.pixels[2, 15, 15] == pytest.approx(3.0)


def test_timestamps_match_fixture(bin_3frames):
    """Task 1.5.5 — timestamps match 50 Hz fraction-of-day spacing."""
    data = DragerBinParser().parse(bin_3frames)
    assert data.timestamps[0] == pytest.approx(0.0)
    assert data.timestamps[1] == pytest.approx(_DT_DAY)
    assert data.timestamps[2] == pytest.approx(2 * _DT_DAY)


def test_timestamps_monotonically_increasing(bin_3frames):
    """Task 1.5.5 — timestamps are strictly increasing."""
    data = DragerBinParser().parse(bin_3frames)
    assert all(data.timestamps[i] < data.timestamps[i + 1] for i in range(2))


def test_roundtrip_via_load_data(bin_3frames):
    """Task 1.5.10 — full pipeline through load_data() orchestrator."""
    from fasteit.parsers.loader import load_data

    data = load_data(bin_3frames)
    assert data.n_frames == 3
    assert data.pixels.shape == (3, 32, 32)
    assert data.pixels[0].mean() == pytest.approx(1.0)
    assert data.timestamps[0] == pytest.approx(0.0)
    assert data.metadata["detected_vendor"] == "draeger"
    assert data.metadata["detected_extension"] == ".bin"
