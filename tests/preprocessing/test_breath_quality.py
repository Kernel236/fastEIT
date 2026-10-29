"""Tests for the breath quality flags."""

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.breath_quality import (
    breath_measures,
    quality_flags,
    ratio_to_local_median,
)

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


# --- flags ---


def flags(signal, fs, breaths, near_invalid=None):
    if near_invalid is None:
        near_invalid = np.zeros(len(breaths), dtype=bool)
    return quality_flags(
        signal,
        fs,
        breaths,
        near_invalid,
        short_seconds=CFG.breath_short_seconds,
        long_seconds=CFG.breath_long_seconds,
        ti_ratio_low=CFG.ti_ratio_low,
        ti_ratio_high=CFG.ti_ratio_high,
        window_breaths=CFG.local_window_breaths,
        min_breaths=CFG.local_min_breaths,
        amplitude_ratio_low=CFG.amplitude_ratio_low,
        amplitude_ratio_high=CFG.amplitude_ratio_high,
        duration_ratio_high=CFG.duration_ratio_high,
    )


def breaths_from_durations(durations, fs=10):
    """Breaths lasting ``durations`` seconds, Ti/Ttot 0.4, every peak of height 1."""
    ends = np.cumsum(np.asarray(durations) * fs).astype(int)
    starts = np.r_[0, ends[:-1]]
    peaks = starts + (0.4 * (ends - starts)).astype(int)
    signal = np.zeros(ends[-1] + 1)
    signal[peaks] = 1.0
    return signal, np.column_stack([starts, peaks, ends])


def test_regular_breathing_has_no_flags():
    signal, breaths = breaths_from_durations([4.0] * 30)
    assert not flags(signal, 10, breaths).any(axis=None)


def test_each_flag_catches_its_problem():
    signal, breaths = breaths_from_durations([4.0] * 30)
    signal[breaths[10, 1]] = 0.3  # a small breath
    breaths[20, 1] = breaths[20, 0] + 30  # peak moved: Ti/Ttot 0.75
    signal[breaths[20, 1]] = 1.0
    near_invalid = np.zeros(len(breaths), dtype=bool)
    near_invalid[25] = True
    result = flags(signal, 10, breaths, near_invalid)
    assert list(np.flatnonzero(result.flag_amplitude_outlier)) == [10]
    assert list(np.flatnonzero(result.flag_suspect_ti_ratio)) == [20]
    assert list(np.flatnonzero(result.flag_invalid_data)) == [25]


def test_two_breaths_counted_as_one_are_a_duration_outlier():
    durations = [4.0] * 30
    durations[10] = 8.5  # e.g. a missed valley
    signal, breaths = breaths_from_durations(durations)
    result = flags(signal, 10, breaths)
    assert list(np.flatnonzero(result.flag_duration_outlier)) == [10]


def test_short_and_long_breaths():
    fs = 10
    signal = np.zeros(400)
    breaths = np.array([[0, 2, 4], [4, 100, 300]])  # 0.4 s and 29.6 s
    signal[[2, 100]] = 1.0
    result = flags(signal, fs, breaths)
    assert list(result.flag_short) == [True, False]
    assert list(result.flag_long) == [False, True]
    # Only 2 breaths: too few for a local median, so no outlier flag.
    assert not result.flag_duration_outlier.any()


def test_no_breaths_give_an_empty_table_with_all_columns():
    result = flags(np.zeros(10), 50, np.empty((0, 3), dtype=int))
    assert len(result) == 0 and "flag_invalid_data" in result.columns
