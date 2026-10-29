"""Tests for the layer 2 containers."""

import pytest

from fasteit.models.eit_recording import RecordingContext, Segment


def test_segment_defaults_to_manual_exclusion():
    s = Segment(10.0, 20.0)
    assert s.category == "exclusion"
    assert s.source == "manual"


def test_segment_rejects_end_before_start():
    with pytest.raises(ValueError, match="must be after start"):
        Segment(20.0, 10.0)


def test_segment_overlaps():
    s = Segment(10.0, 20.0)
    assert s.overlaps(15.0, 25.0)
    assert s.overlaps(5.0, 10.0)  # touching counts
    assert not s.overlaps(21.0, 30.0)


def test_segment_is_immutable():
    s = Segment(10.0, 20.0)
    with pytest.raises(AttributeError):
        s.start = 0.0


def test_recording_context_defaults():
    ctx = RecordingContext(patient_id="PZ01")
    assert ctx.protocol == "tidal"
    assert ctx.position is None
