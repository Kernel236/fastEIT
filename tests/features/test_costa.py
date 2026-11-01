"""Tests for the Costa method on pixels with a known answer."""

import numpy as np
import pytest

from fasteit.features.costa import collapse_and_overdistension, crossing_peep

# Pixel 1 has its best compliance at PEEP 16 and loses half of it at 6; pixel 2 has its
# best at 6 and three times the best compliance of pixel 1 (notebook 07, section 4).
PEEP = np.array([16.0, 6.0])
PIXELS = np.array([[[1.0, 1.5]], [[0.5, 3.0]]])
MASK = np.ones((1, 2), dtype=bool)


def test_two_pixels_give_the_weighted_collapse_and_overdistension():
    table = collapse_and_overdistension(PIXELS, PEEP, MASK)
    # PEEP 6: pixel 1 lost 50 % with weight 1 of 4; PEEP 16: pixel 2 lost 50 %, weight 3.
    np.testing.assert_allclose(table.collapse, [0.0, 12.5])
    np.testing.assert_allclose(table.overdistension, [37.5, 0.0])


def test_pixels_outside_the_mask_are_not_used():
    table = collapse_and_overdistension(PIXELS, PEEP, np.array([[True, False]]))
    np.testing.assert_allclose(table.collapse, [0.0, 50.0])
    np.testing.assert_allclose(table.overdistension, [0.0, 0.0])


def test_a_step_without_compliance_is_nan_and_the_others_are_computed():
    pixels = np.concatenate([PIXELS, np.full((1, 1, 2), np.nan)])
    table = collapse_and_overdistension(pixels, np.r_[PEEP, 4.0], MASK)
    np.testing.assert_allclose(table.collapse[:2], [0.0, 12.5])
    assert table.iloc[2].isna().all()


def test_with_one_step_only_nothing_can_be_compared():
    pixels = np.concatenate([PIXELS[:1], np.full((1, 1, 2), np.nan)])
    assert collapse_and_overdistension(pixels, PEEP, MASK).isna().all().all()


def test_without_any_compliance_everything_is_nan():
    table = collapse_and_overdistension(np.full((2, 1, 2), np.nan), PEEP, MASK)
    assert table.isna().all().all()


@pytest.mark.parametrize(
    "peep, collapse, overdistension, expected",
    [
        ([10, 0], [0, 20], [20, 0], 5.0),  # decremental order, crossing in the middle
        ([0, 5, 10], [20, 10, 0], [0, 10, 20], 5.0),  # a step on the crossing
        ([0, 10], [20, 30], [0, 10], np.nan),  # collapse always higher
        ([0, 10, 20], [20, np.nan, 0], [0, np.nan, 20], 10.0),  # a step without values
        ([0, 5, 10], [5, 0, 3], [0, 0, 0], np.nan),  # the curves touch, do not cross
        ([0, 5, 10], [20, 10, 0], [0, 0, 0], np.nan),  # they meet only at the top
        ([0, 5, 10, 15], [10, 0, 10, 0], [0, 10, 0, 10], 2.5),  # twice: the lowest
    ],
)
def test_crossing_peep(peep, collapse, overdistension, expected):
    result = crossing_peep(np.array(peep, float), np.array(collapse, float),
                           np.array(overdistension, float))  # fmt: skip
    assert result == pytest.approx(expected, nan_ok=True)
