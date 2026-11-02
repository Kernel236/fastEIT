"""Equivalence with eitprocessing v1.8.8 for the steps reimplemented from it.

Skipped when eitprocessing is not installed (it is not a fastEIT dependency) or when
the real test recordings are not available. Evidence: notebooks/05_preprocessing_choices.ipynb.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.breaths import (
    BASELINE_SECONDS,
    INVALID_DATA_MULTIPLIER,
    INVALID_DATA_PERCENTILE,
    INVALID_MARGIN_SECONDS,
    LOW_AMPLITUDE_FRACTION,
    MIN_DURATION_SECONDS,
)
from fasteit.preprocessing.cardiac_filter import (
    MDN_NOTCH_HALF_WIDTH_BPM,
    MDN_ORDER,
    MDN_UPPER_LIMIT_BPM,
    RESPIRATORY_RATE_MAX_BPM,
    RESPIRATORY_RATE_MIN_BPM,
    SECONDS_PER_MINUTE,
    SPECTRUM_WINDOW_SECONDS,
    mdn_filter,
    respiratory_rate,
)

pytest.importorskip("eitprocessing")

from eitprocessing.datahandling.loading import load_eit_data  # noqa: E402
from eitprocessing.features.rate_detection import RateDetection  # noqa: E402
from eitprocessing.filters.mdn import MDNFilter  # noqa: E402

TEST_FILES = Path(__file__).parents[2] / "src/fasteit/test_files"
REAL_BIN_FILES = sorted(TEST_FILES.glob("*.bin"))
needs_real_files = pytest.mark.skipif(
    not REAL_BIN_FILES, reason="real .bin files not available (gitignored)"
)


@pytest.mark.parametrize(
    "fs, respiratory_hz, heart_hz", [(50, 0.5, 1.8), (50, 0.6, 1.0), (20, 0.17, 1.07)]
)
def test_mdn_filter_matches_eitprocessing(fs, respiratory_hz, heart_hz):
    pixels = np.random.default_rng(0).normal(size=(2000, 4, 4))
    reference = MDNFilter(respiratory_rate=respiratory_hz, heart_rate=heart_hz).apply(
        pixels, sample_frequency=fs, axis=0
    )
    ours = mdn_filter(
        pixels,
        fs,
        respiratory_hz,
        heart_hz,
        notch_half_width_hz=MDN_NOTCH_HALF_WIDTH_BPM / SECONDS_PER_MINUTE,
        upper_limit_hz=MDN_UPPER_LIMIT_BPM / SECONDS_PER_MINUTE,
        order=MDN_ORDER,
    )
    np.testing.assert_allclose(ours, reference, atol=1e-10)


@needs_real_files
def test_respiratory_rate_matches_eitprocessing_on_real_recordings():
    for path in REAL_BIN_FILES:
        eit = next(iter(load_eit_data(path, vendor="draeger").eit_data.values()))
        reference, _ = RateDetection("adult").apply(
            eit, suppress_length_warnings=True, suppress_edge_case_warning=True
        )
        global_signal = np.nansum(np.asarray(eit.pixel_impedance, float), axis=(1, 2))
        ours = respiratory_rate(
            global_signal,
            eit.sample_frequency,
            min_bpm=RESPIRATORY_RATE_MIN_BPM,
            max_bpm=RESPIRATORY_RATE_MAX_BPM,
            window_seconds=SPECTRUM_WINDOW_SECONDS,
        )
        assert ours == pytest.approx(reference, abs=1e-6), path.name


def detect_like_fasteit(eit, pixels):
    from fasteit.preprocessing.breaths import detect_breaths

    return detect_breaths(
        pixels.sum(axis=(1, 2)),
        eit.sample_frequency,
        invalid_percentile=INVALID_DATA_PERCENTILE,
        invalid_multiplier=INVALID_DATA_MULTIPLIER,
        baseline_seconds=BASELINE_SECONDS,
        min_duration_seconds=MIN_DURATION_SECONDS,
        low_amplitude_fraction=LOW_AMPLITUDE_FRACTION,
        invalid_margin_seconds=INVALID_MARGIN_SECONDS,
    )


def detect_like_eitprocessing(eit, pixels):
    from eitprocessing.features.breath_detection import BreathDetection

    eit_copy = copy.deepcopy(eit)
    eit_copy.pixel_impedance = pixels
    breaths = BreathDetection().find_breaths(eit_copy.get_summed_impedance()).values
    return breaths_as_frames(breaths, eit.time)


def breaths_as_frames(breaths, time):
    """eitprocessing Breath objects -> (n, 3) frame indices like fastEIT."""
    frames = [
        np.searchsorted(time, [b.start_time, b.middle_time, b.end_time])
        for b in breaths
    ]
    return np.array(frames, dtype=int).reshape(-1, 3)


def synthetic_global_signal(case, fs=50, seconds=180):
    """18/min breathing with a slow drift and noise, plus the requested problem."""
    t = np.arange(0, seconds, 1 / fs)
    x = np.sin(2 * np.pi * 0.3 * t) + 0.3 * t / seconds
    x += 0.05 * np.random.default_rng(0).normal(size=t.size)
    if case == "spike":
        x[4000:4003] += 40
    if case == "small breaths":
        x[(t % 20) < 3.3] *= 0.1  # one breath in six is a tenth of the others
    return t, x


@pytest.mark.parametrize("case", ["clean", "spike", "small breaths"])
def test_breaths_match_eitprocessing_on_a_synthetic_signal(case):
    from eitprocessing.datahandling.continuousdata import ContinuousData
    from eitprocessing.features.breath_detection import BreathDetection

    from fasteit.preprocessing.breaths import detect_breaths

    fs = 50
    t, x = synthetic_global_signal(case, fs)
    global_signal = ContinuousData(
        label="global",
        name="global",
        unit="a.u.",
        category="impedance",
        time=t,
        values=x,
        sample_frequency=fs,
    )
    reference = BreathDetection().find_breaths(global_signal).values

    ours, near_invalid = detect_breaths(
        x,
        fs,
        invalid_percentile=INVALID_DATA_PERCENTILE,
        invalid_multiplier=INVALID_DATA_MULTIPLIER,
        baseline_seconds=BASELINE_SECONDS,
        min_duration_seconds=MIN_DURATION_SECONDS,
        low_amplitude_fraction=LOW_AMPLITUDE_FRACTION,
        invalid_margin_seconds=INVALID_MARGIN_SECONDS,
    )
    assert near_invalid.any() == (case == "spike")
    np.testing.assert_array_equal(ours[~near_invalid], breaths_as_frames(reference, t))


@needs_real_files
def test_breaths_match_eitprocessing_on_real_recordings():
    from fasteit.preprocessing.cardiac_filter import moving_average

    for path in REAL_BIN_FILES:
        eit = next(iter(load_eit_data(path, vendor="draeger").eit_data.values()))
        pixels = moving_average(
            np.asarray(eit.pixel_impedance, float), eit.sample_frequency, 0.1
        )
        ours, near_invalid = detect_like_fasteit(eit, pixels)
        np.testing.assert_array_equal(
            ours[~near_invalid], detect_like_eitprocessing(eit, pixels)
        )


@needs_real_files
def test_breaths_near_a_spike_are_flagged_where_eitprocessing_removes_them():
    eit = next(
        iter(load_eit_data(REAL_BIN_FILES[-1], vendor="draeger").eit_data.values())
    )
    pixels = np.asarray(eit.pixel_impedance, float).copy()
    pixels[3000:3003] += 50 * np.abs(pixels).max()  # a disconnection-like spike
    ours, near_invalid = detect_like_fasteit(eit, pixels)
    assert near_invalid.any()
    np.testing.assert_array_equal(
        ours[~near_invalid], detect_like_eitprocessing(eit, pixels)
    )


@pytest.mark.slow  # about 35 s: eitprocessing computes every map again
@needs_real_files
def test_watershed_mask_matches_eitprocessing_on_real_recordings():
    from eitprocessing.roi.watershed import WatershedLungspace

    from fasteit.preprocessing.cardiac_filter import moving_average
    from fasteit.preprocessing.lung_mask import watershed_mask

    cfg = PreprocessingConfig()
    for path in REAL_BIN_FILES:
        eit = next(iter(load_eit_data(path, vendor="draeger").eit_data.values()))
        pixels = moving_average(
            np.asarray(eit.pixel_impedance, float), eit.sample_frequency, 0.1
        )
        breaths, _ = detect_like_fasteit(eit, pixels)
        ours, tiv_map, amplitude_map = watershed_mask(
            pixels,
            pixels.sum(axis=(1, 2)),
            breaths,
            np.ones(len(breaths), dtype=bool),
            cfg.lung_mask_threshold_fraction,
        )
        eit_copy = copy.deepcopy(eit)
        eit_copy.pixel_impedance = pixels
        captured = {}
        reference = WatershedLungspace(
            threshold_fraction=cfg.lung_mask_threshold_fraction
        ).apply(eit_copy, captures=captured)
        np.testing.assert_allclose(tiv_map, captured["mean tiv"].values, atol=1e-10)
        np.testing.assert_allclose(
            amplitude_map, captured["mean amplitude"].values, atol=1e-10
        )
        np.testing.assert_array_equal(ours, ~np.isnan(reference.mask), path.name)


@pytest.mark.parametrize("seed", range(5))
def test_small_region_removal_matches_eitprocessing(seed):
    from eitprocessing.roi import PixelMask
    from eitprocessing.roi.filter_by_size import FilterROIBySize

    from fasteit.preprocessing.lung_mask import remove_small_regions

    mask = np.random.default_rng(seed).random((32, 32)) > 0.55
    reference = FilterROIBySize(min_region_size=10).apply(
        PixelMask(np.where(mask, 1.0, np.nan))
    )
    np.testing.assert_array_equal(
        remove_small_regions(mask, 10), ~np.isnan(reference.mask)
    )
