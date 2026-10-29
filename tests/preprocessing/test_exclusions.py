"""Tests for the exclusions."""

import numpy as np
import pytest

from fasteit.models.eit_recording import Segment
from fasteit.preprocessing.exclusions import (
    breaths_in_segments,
    fill_missing_frames,
    frames_to_segments,
    merge_overlapping,
    missing_frames,
    widen,
)

FS = 10
TIME = np.arange(100) / FS  # 10 s


# --- device exclusions ---


def test_runs_of_bad_frames_become_segments():
    bad = np.zeros(100, dtype=bool)
    bad[[20, 21, 22, 50]] = True
    segments = frames_to_segments(TIME, bad, FS, "timing_error")
    np.testing.assert_allclose(
        [(s.start, s.end) for s in segments], [(2.0, 2.3), (5.0, 5.1)]
    )
    assert all(s.source == "device" and s.label == "timing_error" for s in segments)


def test_no_bad_frames_no_segments():
    assert frames_to_segments(TIME, np.zeros(100, dtype=bool), FS, "x") == []


def test_a_frame_is_missing_only_if_a_pixel_has_a_value_elsewhere():
    pixels = np.ones((5, 2, 2))
    pixels[:, 0, 0] = np.nan  # outside the belt: never has a value
    pixels[3, 1, 1] = np.nan  # a real gap
    assert missing_frames(pixels).tolist() == [False, False, False, True, False]


def test_missing_frames_take_the_nearest_valid_frame():
    pixels = np.arange(6, dtype=float)[:, None, None] * np.ones((1, 2, 2))
    pixels[[2, 3]] = np.nan
    filled = fill_missing_frames(pixels, missing_frames(pixels))
    assert filled[:, 0, 0].tolist() == [0, 1, 1, 4, 4, 5]


# --- margin and merging ---


def test_widen_adds_the_margin_inside_the_recording():
    segments = widen([Segment(0.2, 3.0), Segment(9.0, 9.8)], margin=0.5, duration=10)
    assert [(s.start, s.end) for s in segments] == [(0.0, 3.5), (8.5, 10.0)]


def test_overlapping_segments_of_the_same_source_are_joined():
    segments = merge_overlapping(
        [
            Segment(5, 8, label="movement"),
            Segment(1, 3, label="artifact"),
            Segment(2, 6, label="artifact"),
        ]
    )
    assert len(segments) == 1
    assert (segments[0].start, segments[0].end) == (1, 8)
    assert segments[0].label == "artifact, movement"


def test_touching_and_unlabelled_segments_are_joined_without_empty_labels():
    segments = merge_overlapping(
        [
            Segment(1, 3),
            Segment(3, 5, label="artifact"),
            Segment(4, 6, label="artifact"),
        ]
    )
    assert len(segments) == 1 and segments[0].label == "artifact"


def test_an_exclusion_after_the_end_of_the_recording_is_an_error():
    with pytest.raises(ValueError, match="after the end"):
        widen([Segment(20, 30)], margin=0.5, duration=10)


def test_manual_and_device_segments_are_never_joined():
    segments = merge_overlapping(
        [Segment(1, 3, label="artifact"), Segment(2, 4, source="device")]
    )
    assert [s.source for s in segments] == ["manual", "device"]


# --- breaths ---


def test_breaths_touching_an_exclusion_are_excluded_with_its_label():
    starts, ends = np.array([0.0, 4.0, 8.0]), np.array([4.0, 8.0, 12.0])
    segments = [Segment(5, 6, label="cough"), Segment(7.5, 9, label="artifact")]
    excluded, labels = breaths_in_segments(starts, ends, segments)
    assert excluded.tolist() == [False, True, True]
    assert labels == [None, "cough, artifact", "artifact"]


def test_joined_labels_are_not_repeated_and_missing_labels_are_named():
    starts, ends = np.array([0.0, 4.0]), np.array([4.0, 8.0])
    segments = [Segment(1, 2, label="artifact, cough"), Segment(2, 3, label="cough")]
    segments.append(Segment(5, 6))
    assert breaths_in_segments(starts, ends, segments)[1] == [
        "artifact, cough",
        "unlabelled",
    ]
