"""Tests for the regions of interest."""

import numpy as np
import pytest

from fasteit.preprocessing.regions import lung_regions


def lungs_from_row(first_row, height):
    """Two rectangular lungs, from ``first_row`` for ``height`` rows."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[first_row : first_row + height, 4:14] = True  # patient right
    mask[first_row : first_row + height, 18:28] = True  # patient left
    return mask


def test_halves_and_layers_are_cut_on_the_lung_not_on_the_image():
    regions = lung_regions(lungs_from_row(first_row=10, height=8))
    assert np.flatnonzero(regions["ventral"].any(axis=1)).tolist() == [10, 11, 12, 13]
    assert np.flatnonzero(regions["dorsal"].any(axis=1)).tolist() == [14, 15, 16, 17]
    for k, rows in enumerate([[10, 11], [12, 13], [14, 15], [16, 17]], start=1):
        assert np.flatnonzero(regions[f"layer_{k}"].any(axis=1)).tolist() == rows


def test_lung_of_odd_height_gives_the_middle_row_to_the_ventral_half():
    regions = lung_regions(lungs_from_row(first_row=0, height=5))
    assert regions["ventral"].any(axis=1).sum() == 3
    assert regions["dorsal"].any(axis=1).sum() == 2


def test_image_left_is_patient_right():
    regions = lung_regions(lungs_from_row(first_row=10, height=8))
    assert regions["right"][:, :16].sum() == regions["right"].sum() == 80
    assert regions["left"][:, 16:].sum() == regions["left"].sum() == 80


@pytest.mark.parametrize(
    "family",
    [
        ["ventral", "dorsal"],
        ["right", "left"],
        ["layer_1", "layer_2", "layer_3", "layer_4"],
        ["ventral_right", "ventral_left", "dorsal_right", "dorsal_left"],
    ],
)
def test_each_family_splits_the_lung_without_overlap(family):
    mask = lungs_from_row(first_row=7, height=13)
    regions = lung_regions(mask)
    total = sum(regions[name].astype(int) for name in family)
    np.testing.assert_array_equal(total, mask.astype(int))


def test_empty_mask_is_an_error():
    with pytest.raises(ValueError, match="Empty"):
        lung_regions(np.zeros((32, 32), dtype=bool))
