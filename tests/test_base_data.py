"""Tests for BaseData dataclass (Task 0.5.2)."""

import numpy as np

from fasteit.models.base_data import BaseData
from fasteit.models.raw_impedance_data import RawImpedanceData


def test_base_data_defaults():
    d = BaseData()
    assert d.n_frames == 0
    assert d.duration == 0.0
    assert d.fs is None
    assert d.filename == ""
    assert d.file_format == ""
    assert d.vendor == ""
    assert d.metadata == {}


def test_base_data_metadata_isolation():
    """Two instances must not share the same metadata dict."""
    d1 = BaseData()
    d2 = BaseData()
    d1.metadata["key"] = "value"
    assert "key" not in d2.metadata


# ── RawImpedanceData ──────────────────────────────────────────────────────────


def test_raw_impedance_data_n_frames():
    meas = np.zeros((10, 208), dtype=np.float32)
    d = RawImpedanceData(measurements=meas, fs=20.0)
    assert d.n_frames == 10


def test_raw_impedance_data_duration():
    meas = np.zeros((100, 208), dtype=np.float32)
    d = RawImpedanceData(measurements=meas, fs=20.0)
    assert d.duration == 100 / 20.0


def test_raw_impedance_data_no_measurements():
    d = RawImpedanceData()
    assert d.n_frames == 0
