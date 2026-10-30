"""Tests for the PEEP steps and the breaths analysed."""

import numpy as np
import pytest

from fasteit.config import SegmentationConfig
from fasteit.models.eit_recording import Segment
from fasteit.segmentation.peep_steps import (
    analysed_breaths,
    find_steps,
    step_of_each_breath,
    steps_from_segments,
)

CFG = SegmentationConfig()


def steps(peep):
    return find_steps(
        np.asarray(peep, dtype=float),
        CFG.step_threshold,
        CFG.step_min_breaths,
        CFG.step_agreement,
    )


# --- steps from the pressure ---


def test_decremental_steps_of_two():
    peep = np.repeat([16.0, 14.0, 12.0, 10.0], 8) + np.random.default_rng(0).normal(
        0, 0.1, 32
    )
    assert steps(peep).tolist() == [0, 8, 16, 24]


def test_a_single_odd_breath_does_not_open_a_step():
    peep = np.full(20, 6.0)
    peep[10] = 9.0  # e.g. an end-expiratory hold
    assert steps(peep).tolist() == [0]


def test_two_odd_breaths_in_a_row_are_not_enough_either():
    peep = np.full(20, 6.0)
    peep[[10, 11]] = 9.0
    assert steps(peep).tolist() == [0]


def test_breaths_that_disagree_do_not_open_a_step():
    peep = np.r_[np.full(10, 10.0), [12.0, 14.5, 12.0], np.full(10, 10.0)]
    assert steps(peep).tolist() == [0]


def test_incremental_steps_and_steps_of_one_with_a_lower_threshold():
    peep = np.repeat([4.0, 5.0, 6.0], 6)
    assert find_steps(peep, 0.5, 3, 0.5).tolist() == [0, 6, 12]


def test_breaths_without_peep_take_the_next_one():
    peep = np.r_[np.full(6, 10.0), [np.nan], np.full(6, 8.0)]
    assert steps(peep).tolist() == [0, 6]


def test_no_peep_at_all_is_an_error():
    with pytest.raises(ValueError, match="No breath has a PEEP"):
        steps(np.full(5, np.nan))


def test_step_of_each_breath():
    assert step_of_each_breath(np.array([0, 3, 5]), 7).tolist() == [0, 0, 0, 1, 1, 2, 2]


# --- steps placed by hand ---


def peep_step(start, end, peep):
    return Segment(start, end, category="peep_step", parameters={"peep": peep})


def test_steps_placed_by_hand_take_the_breaths_inside_them():
    starts = np.arange(0, 60, 4.0)
    step, levels = steps_from_segments(
        starts, starts + 4, [peep_step(30, 50, 8), peep_step(4, 21, 10)]
    )
    assert levels.tolist() == [10.0, 8.0]  # numbered in time order
    assert np.flatnonzero(step == 0).tolist() == [1, 2, 3, 4]  # breaths 4-8 ... 16-20
    assert np.flatnonzero(step == 1).tolist() == [8, 9, 10, 11]
    assert (step[[0, 5, 6, 7, 12, 13, 14]] == -1).all()


@pytest.mark.parametrize(
    "segments, message",
    [
        ([Segment(0, 10, label="cough")], "not a PEEP step"),
        ([Segment(0, 10, category="peep_step")], "not a PEEP step"),
        ([peep_step(0, 20, 10), peep_step(15, 30, 8)], "overlap"),
    ],
)
def test_unusable_steps_placed_by_hand_are_rejected(segments, message):
    with pytest.raises(ValueError, match=message):
        steps_from_segments(np.arange(10.0), np.arange(1.0, 11.0), segments)


# --- breaths analysed ---


def test_first_seconds_are_skipped_then_the_last_valid_breaths_are_kept():
    starts = np.arange(0, 80, 4.0)  # one step of 20 breaths, every 4 s
    valid = np.ones(20, dtype=bool)
    valid[18] = False
    analysed = analysed_breaths(starts, np.zeros(20, dtype=int), valid, 15.0, 10)
    assert np.flatnonzero(analysed).tolist() == [9, 10, 11, 12, 13, 14, 15, 16, 17, 19]


def test_a_short_step_keeps_what_is_left_after_the_skip():
    starts = np.arange(0, 40, 4.0)  # 10 breaths over 40 s
    analysed = analysed_breaths(
        starts, np.zeros(10, dtype=int), np.ones(10, dtype=bool), 15.0, 10
    )
    assert np.flatnonzero(analysed).tolist() == [4, 5, 6, 7, 8, 9]  # from 16 s on


def test_each_step_is_handled_on_its_own_and_breaths_without_step_are_never_analysed():
    starts = np.arange(0, 120, 4.0)
    step = np.repeat([0, 1, -1], 10)
    analysed = analysed_breaths(starts, step, np.ones(30, dtype=bool), 15.0, 3)
    assert np.flatnonzero(analysed).tolist() == [7, 8, 9, 17, 18, 19]
