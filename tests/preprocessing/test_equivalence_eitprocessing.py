"""Equivalence with eitprocessing v1.8.8 for the steps reimplemented from it.

Skipped when eitprocessing is not installed (it is not a fastEIT dependency) or when
the real test recordings are not available. Evidence: notebooks/05_preprocessing_choices.ipynb.
"""

from pathlib import Path

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.preprocessing.cardiac_filter import (
    SECONDS_PER_MINUTE,
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
    cfg = PreprocessingConfig()
    ours = mdn_filter(
        pixels,
        fs,
        respiratory_hz,
        heart_hz,
        notch_half_width_hz=cfg.mdn_notch_half_width_bpm / SECONDS_PER_MINUTE,
        upper_limit_hz=cfg.mdn_upper_limit_bpm / SECONDS_PER_MINUTE,
        order=cfg.mdn_order,
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
        cfg = PreprocessingConfig()
        ours = respiratory_rate(
            global_signal,
            eit.sample_frequency,
            min_bpm=cfg.respiratory_rate_min_bpm,
            max_bpm=cfg.respiratory_rate_max_bpm,
            window_seconds=cfg.spectrum_window_seconds,
        )
        assert ours == pytest.approx(reference, abs=1e-6), path.name
