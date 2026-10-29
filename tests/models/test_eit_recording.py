"""Tests for the layer 2 containers."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.models.eit_recording import EITRecording, RecordingContext, Segment
from fasteit.models.reconstructed_data import ReconstructedFrameData


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


# --- EITRecording ---


def small_recording():
    breaths = pd.DataFrame({"breath_id": [0, 1], "is_valid": [True, False]})
    return EITRecording(
        raw=ReconstructedFrameData(fs=50.0),
        config=PreprocessingConfig(),
        context=RecordingContext(patient_id="PZ01"),
        exclusions=(),
        time=np.arange(4) / 50,
        rates={"respiratory_rate_hz": 0.3, "heart_rate_hz": None},
        pixels=np.zeros((4, 2, 2), dtype=np.float32),
        global_signal=np.zeros(4),
        breaths=breaths,
        lung_mask=np.ones((2, 2), dtype=bool),
        tiv_map=np.ones((2, 2)),
        roi_masks={"ventral": np.ones((2, 2), dtype=bool)},
        roi_signals=pd.DataFrame({"ventral": np.zeros(4)}),
        log=("done",),
    )


def test_recording_arrays_are_read_only():
    recording = small_recording()
    with pytest.raises(ValueError, match="read-only"):
        recording.pixels[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        recording.roi_masks["ventral"][0, 0] = False


def test_recording_fields_cannot_be_replaced():
    with pytest.raises(dataclasses.FrozenInstanceError):
        small_recording().lung_mask = None


def test_recording_gives_fs_and_valid_breaths():
    recording = small_recording()
    assert recording.fs == 50.0
    assert recording.valid_breaths.breath_id.tolist() == [0]
