"""Tests for the indices of one breath, on tidal images with a known answer."""

import numpy as np
import pytest

from fasteit.features.breath_indices import (
    centre_of_ventilation,
    global_inhomogeneity,
    tidal_images,
)


def lungs(ventral=1.0, dorsal=1.0, right=1.0):
    """Tidal image: a lung area of rows 8-23 and columns 4-27, NaN outside.

    Rows 16-23 are the dorsal half; columns 4-15 are the patient's right (image left).
    """
    image = np.full((32, 32), np.nan)
    image[8:16, 4:28] = ventral
    image[16:24, 4:28] = dorsal
    image[8:24, 4:16] *= right
    return image


def test_tidal_images_are_peak_minus_start_inside_the_lung():
    pixels = np.zeros((4, 2, 2))
    pixels[2] = [[3.0, 5.0], [7.0, 9.0]]
    mask = np.array([[True, False], [True, True]])
    images = tidal_images(pixels, np.array([0]), np.array([2]), mask)
    np.testing.assert_array_equal(images[0], [[3.0, np.nan], [7.0, 9.0]])


def test_homogeneous_ventilation_has_no_inhomogeneity_and_a_central_centre():
    assert global_inhomogeneity(lungs()) == 0.0
    assert centre_of_ventilation(lungs()) == pytest.approx((0.5, 0.5))


def test_more_dorsal_ventilation_moves_the_centre_dorsal():
    image = lungs(dorsal=3.0)
    # Half the pixels at 1, half at 3: median 2, sum |pixel - 2| = N, sum = 2 N.
    assert global_inhomogeneity(image) == pytest.approx(0.5)
    # Mean row (1 x 11.5 + 3 x 19.5) / 4 = 17.5, over the last row 31.
    vd, rl = centre_of_ventilation(image)
    assert vd == pytest.approx(17.5 / 31) and rl == pytest.approx(0.5)


def test_more_ventilation_in_the_right_lung_moves_the_centre_to_0():
    # Mean column (3 x 9.5 + 1 x 21.5) / 4 = 12.5, over the last column 31.
    vd, rl = centre_of_ventilation(lungs(right=3.0))
    assert vd == pytest.approx(0.5) and rl == pytest.approx(12.5 / 31)
    assert global_inhomogeneity(lungs(right=3.0)) == pytest.approx(0.5)


def test_a_lung_without_tidal_variation_gives_nan():
    flat = lungs() * 0
    assert np.isnan(global_inhomogeneity(flat))
    assert np.isnan(centre_of_ventilation(flat)).all()
