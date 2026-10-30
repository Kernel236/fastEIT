"""Tests for the PEEP steps and the breaths analysed."""

import numpy as np
import pytest

from fasteit.config import SegmentationConfig
from fasteit.segmentation.peep_steps import (
    find_steps,
    step_of_each_breath,
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
