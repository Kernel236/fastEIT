"""Tests for DragerBinParser using a synthetic .bin file."""

from pathlib import Path

import numpy as np

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


def test_parse_float_sentinel_replaced_with_nan(tmp_path):
    frames = np.zeros(3, dtype=FRAME_BASE_DTYPE)
    dt_day = 1.0 / (50.0 * 86400.0)
    frames["ts"] = np.arange(3) * dt_day
    frames["pixels"][1, 0, 0] = -1000.0  # float sentinel value
    path = tmp_path / "sentinel.bin"
    frames.tofile(path)

    data = DragerBinParser().parse(path)
    assert np.isnan(data.pixels[1, 0, 0])
