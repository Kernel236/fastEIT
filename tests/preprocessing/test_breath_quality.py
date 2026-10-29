"""Tests for the breath quality flags."""

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.breath_quality import breath_measures, ratio_to_local_median

CFG = PreprocessingConfig()


# --- measures and local ratios ---


def test_measures_of_a_hand_built_breath():
    # Valley 0 at 0, peak at frame 20 (value 4), valley 2 at frame 50 (value 2).
    signal = np.zeros(51)
    signal[20], signal[50] = 4.0, 2.0
    measures = breath_measures(signal, fs=10, breaths=np.array([[0, 20, 50]]))
    assert measures.duration_s[0] == pytest.approx(5.0)
    assert measures.ti_ratio[0] == pytest.approx(0.4)
    assert measures.amplitude[0] == pytest.approx(3.0)  # rise 4, fall 2


def test_measures_of_no_breaths():
    measures = breath_measures(np.zeros(10), 50, np.empty((0, 3), dtype=int))
    assert len(measures) == 0
    assert list(measures.columns) == ["duration_s", "ti_ratio", "amplitude"]


def test_ratio_to_local_median_finds_the_odd_breath():
    values = np.ones(30)
    values[15] = 3.0
    ratios = ratio_to_local_median(values, window_breaths=21, min_breaths=5)
    assert ratios[15] == pytest.approx(3.0)
    assert np.allclose(np.delete(ratios, 15), 1.0)


def test_ratio_to_local_median_follows_a_change_of_level():
    # e.g. tidal impedance halves after a PEEP step: no breath is an outlier.
    values = np.r_[np.full(40, 2.0), np.full(40, 1.0)]
    ratios = ratio_to_local_median(values, window_breaths=21, min_breaths=5)
    assert np.all((ratios > 0.5) & (ratios < 2.0))


def test_ratio_is_nan_with_too_few_breaths():
    ratios = ratio_to_local_median(np.ones(3), window_breaths=21, min_breaths=5)
    assert np.all(np.isnan(ratios))
