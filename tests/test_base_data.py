"""Tests for BaseData dataclass (Task 0.5.2)."""

from fasteit.models.base_data import BaseData


def test_base_data_defaults():
    d = BaseData()
    assert d.n_frames == 0
    assert d.duration == 0.0
    assert d.fs == 50.0
    assert d.filename == ""
    assert d.file_format == ""
    assert d.metadata == {}


def test_base_data_metadata_isolation():
    """Two instances must not share the same metadata dict."""
    d1 = BaseData()
    d2 = BaseData()
    d1.metadata["key"] = "value"
    assert "key" not in d2.metadata
