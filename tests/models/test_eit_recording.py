"""Tests for the layer 2 containers."""

import pytest

from fasteit.models.eit_recording import RecordingContext, Segment


def test_segment_defaults_to_manual_exclusion():
    s = Segment(10.0, 20.0)
    assert s.category == "exclusion"
    assert s.source == "manual"


@pytest.mark.parametrize(
    "start, end, message",
    [
        (20.0, 10.0, "must be after start"),
        (10.0, 10.0, "must be after start"),
        (float("nan"), 10.0, "finite"),
        (0.0, float("inf"), "finite"),
        (-1.0, 10.0, ">= 0"),
    ],
)
def test_segment_rejects_invalid_times(start, end, message):
    with pytest.raises(ValueError, match=message):
        Segment(start, end)


def test_segment_overlaps():
    s = Segment(10.0, 20.0)
    assert s.overlaps(15.0, 25.0)
    assert s.overlaps(5.0, 10.0)  # touching counts
    assert s.overlaps(20.0, 25.0)  # touching at the end counts too
    assert not s.overlaps(21.0, 30.0)


def test_segment_is_immutable():
    s = Segment(10.0, 20.0)
    with pytest.raises(AttributeError):
        s.start = 0.0


def test_recording_context_defaults():
    ctx = RecordingContext(patient_id="PZ01")
    assert ctx.protocol == "tidal"
    assert ctx.position is None


def test_segments_are_hashable_and_parameters_not_shared():
    a, b = Segment(0.0, 1.0), Segment(2.0, 3.0)
    assert len({a, b, Segment(0.0, 1.0)}) == 2
    assert a.parameters is not b.parameters
