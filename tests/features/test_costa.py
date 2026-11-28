"""Tests for the Costa method on pixels with a known answer."""

import numpy as np
import pytest

from fasteit.features.costa import (
    collapse_and_overdistension,
    costa_weight,
    crossing_peep,
)

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


def test_a_pixel_no_more_ventilated_loses_at_most_100_percent():
    # Pixel 1 is best at PEEP 16 and its TIV turns negative at PEEP 6 (noise around 0).
    pixels = np.array([[[1.0, 1.5]], [[-0.2, 3.0]]])
    table = collapse_and_overdistension(pixels, PEEP, MASK)
    np.testing.assert_allclose(table.collapse, [0.0, 100 * 1 / 4])


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


def test_costa_weight_is_the_share_of_the_best_compliance_of_the_lung():
    # Best compliance 1 for pixel 1 and 3 for pixel 2.
    assert costa_weight(PIXELS, np.array([[True, False]]), MASK) == pytest.approx(0.25)
    assert costa_weight(PIXELS, np.array([[False, True]]), MASK) == pytest.approx(0.75)


def test_on_a_partition_the_regions_weighted_by_costa_weight_give_the_lung():
    rng = np.random.default_rng(0)
    pixels = rng.uniform(-0.2, 2.0, size=(5, 4, 4))  # some pixels never ventilated
    pixels[2] = np.nan  # a step without a driving pressure
    peep = np.array([12.0, 10.0, 8.0, 6.0, 4.0])
    lung = np.ones((4, 4), dtype=bool)
    lung[0, 0] = False
    upper = lung.copy()
    upper[2:] = False
    lower = lung & ~upper
    recomposed = sum(
        costa_weight(pixels, region, lung)
        * collapse_and_overdistension(pixels, peep, region)
        for region in (upper, lower)
    )
    whole = collapse_and_overdistension(pixels, peep, lung)
    np.testing.assert_allclose(recomposed.to_numpy(), whole.to_numpy())


def test_a_region_without_ventilated_pixels_has_no_values_and_weight_0():
    pixels = PIXELS.copy()
    pixels[:, 0, 0] = -0.1  # pixel 1 never ventilated
    region = np.array([[True, False]])
    assert collapse_and_overdistension(pixels, PEEP, region).isna().all().all()
    assert costa_weight(pixels, region, MASK) == 0


def test_costa_weight_needs_two_steps_with_compliance():
    pixels = np.concatenate([PIXELS[:1], np.full((1, 1, 2), np.nan)])
    assert np.isnan(costa_weight(pixels, MASK, MASK))
    assert np.isnan(costa_weight(np.full((2, 1, 2), -0.1), MASK, MASK))


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
