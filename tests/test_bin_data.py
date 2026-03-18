"""Tests for BinData dataclass (Task 0.5.2).

Note: FRAME_BASE_DTYPE is defined inline here matching the corrected architecture.
It will be replaced by `from fasteit.dtypes import FRAME_BASE_DTYPE` in Task 1.2.1.
"""

import numpy as np
import pytest

from fasteit.models.bin_data import BinData

# ── Minimal dtype matching corrected fastEIT_layerdata_class_architecture.md ──
# 4 + 4 + 4 + 4096 + 8 + 30 + 4 + 208 = 4358 bytes
_FRAME_BASE_DTYPE = np.dtype(
    [
        ("ts1", "<f4"),
        ("ts2", "<f4"),
        ("dummy", "<f4"),
        ("pixels", "<f4", (32, 32)),
        ("minmax_event", "<i4", (2,)),
        ("event_text", "S30"),
        ("timing_error", "<i4"),
        ("medibus_or_padding", "u1", (208,)),
    ]
)

assert _FRAME_BASE_DTYPE.itemsize == 4358, (
    f"Dtype size mismatch: expected 4358, got {_FRAME_BASE_DTYPE.itemsize}"
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture()
def five_frames() -> np.ndarray:
    frames = np.zeros(5, dtype=_FRAME_BASE_DTYPE)
    frames["ts1"] = np.arange(5) * 0.05  # 0.00, 0.05, 0.10, 0.15, 0.20
    frames["pixels"][:, 16, 16] = 1.0  # one pixel lit per frame
    return frames


@pytest.fixture()
def bin_data(five_frames) -> BinData:
    return BinData(
        frames=five_frames,
        filename="test.bin",
        file_format="bin",
    )


# ── Tests: __post_init__ ─────────────────────────────────────────────────


def test_n_frames(bin_data):
    assert bin_data.n_frames == 5


def test_duration(bin_data):
    assert bin_data.duration == pytest.approx(0.10)  # 5 / 50.0 Hz


def test_no_frames_gives_zero():
    d = BinData()
    assert d.n_frames == 0
    assert d.duration == 0.0


# ── Tests: property accessors ────────────────────────────────────────────


def test_timestamps_shape(bin_data):
    assert bin_data.timestamps.shape == (5,)


def test_timestamps_values(bin_data):
    assert bin_data.timestamps[2] == pytest.approx(0.10)


def test_pixels_shape(bin_data):
    assert bin_data.pixels.shape == (5, 32, 32)


def test_event_texts_shape(bin_data):
    assert bin_data.event_texts.shape == (5,)


# ── Tests: derived signals ────────────────────────────────────────────────


def test_global_signal_shape(bin_data):
    assert bin_data.global_signal.shape == (5,)


def test_global_signal_value(bin_data):
    # only pixel (16,16) is lit with value 1.0 → sum = 1.0 per frame
    assert bin_data.global_signal[0] == pytest.approx(1.0)


def test_roi_signals_shape(bin_data):
    assert bin_data.roi_signals.shape == (5, 4)


def test_roi_signals_mid_dorsal(bin_data):
    # pixel row 16 is in ROI 2 (rows 16-23)
    rois = bin_data.roi_signals
    assert rois[:, 2].sum() == pytest.approx(5.0)  # 5 frames × 1.0
    assert rois[:, 0].sum() == pytest.approx(0.0)
    assert rois[:, 1].sum() == pytest.approx(0.0)
    assert rois[:, 3].sum() == pytest.approx(0.0)


def test_roi_signal_single(bin_data):
    sig = bin_data.roi_signal(2)
    assert sig.shape == (5,)
    assert sig.sum() == pytest.approx(5.0)


def test_roi_signal_invalid_index(bin_data):
    with pytest.raises(ValueError, match="ROI deve essere 0-3"):
        bin_data.roi_signal(4)

    with pytest.raises(ValueError, match="ROI deve essere 0-3"):
        bin_data.roi_signal(-1)


# ── Tests: custom fs ─────────────────────────────────────────────────────


def test_custom_sampling_rate(five_frames):
    d = BinData(frames=five_frames, fs=50.0)
    assert d.duration == pytest.approx(5 / 50.0)
