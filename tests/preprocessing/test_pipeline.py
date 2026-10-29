"""Tests for preprocess()."""

from pathlib import Path

import numpy as np
import pytest

from fasteit.config import PreprocessingConfig
from fasteit.models.eit_recording import Segment
from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.loader import load_data
from fasteit.preprocessing.pipeline import preprocess, recording_time

from .test_breath_quality import NOTEBOOK_COUNTS, TEST_FILES

FS = 20.0
SECONDS = 90


def synthetic_raw(vendor="timpel", seconds=SECONDS, breathing_hz=0.25):
    """Two lungs breathing at 15/min in a 32 x 32 image, with a little noise.

    Time is in seconds (as Timpel) unless the vendor is "draeger" (fraction of day).
    """
    t = np.arange(0, seconds, 1 / FS)
    rows, cols = np.mgrid[0:32, 0:32]
    lungs = np.exp(-(((rows - 14) / 6) ** 2 + ((cols - 9) / 4) ** 2))
    lungs += np.exp(-(((rows - 14) / 6) ** 2 + ((cols - 22) / 4) ** 2))
    breathing = np.sin(np.pi * breathing_hz * t) ** 2
    pixels = breathing[:, None, None] * lungs
    pixels += 0.01 * np.random.default_rng(0).normal(size=pixels.shape)

    frames = np.zeros(
        len(t),
        dtype=[("ts", "f8"), ("pixels", "f4", (32, 32)), ("timing_error", "i4")],
    )
    frames["ts"] = t / 86400 + 0.5 if vendor == "draeger" else t
    frames["pixels"] = pixels
    raw = ReconstructedFrameData(fs=FS, vendor=vendor, frames=frames)
    return raw, lungs


# --- time ---


def test_draeger_time_crosses_midnight():
    raw, _ = synthetic_raw(vendor="draeger", seconds=4)
    raw.frames["ts"] = (np.arange(80) / FS + 86398) / 86400 % 1  # from 23:59:58
    time = recording_time(raw)
    np.testing.assert_allclose(time, np.arange(80) / FS, atol=1e-6)


def test_timpel_time_is_already_in_seconds():
    raw, _ = synthetic_raw(seconds=4)
    raw.frames["ts"] += 2.0
    np.testing.assert_allclose(recording_time(raw), np.arange(80) / FS)


# --- whole pipeline on a synthetic recording ---


def test_breaths_mask_and_regions_of_a_regular_recording():
    raw, lungs = synthetic_raw()
    recording = preprocess(raw)
    assert 19 <= len(recording.breaths) <= 22  # 22 breaths, edges may be dropped
    assert recording.breaths.is_valid.all()
    assert recording.lung_mask[lungs > 0.5].all()
    assert not recording.lung_mask[lungs < 0.01].any()
    assert set(recording.roi_signals.columns) == set(recording.roi_masks)
    assert recording.rates["respiratory_rate_hz"] == pytest.approx(0.25, abs=0.01)
    assert recording.rates["heart_rate_hz"] is None


def test_without_heart_rate_the_log_says_the_heart_is_not_removed():
    recording = preprocess(synthetic_raw()[0])
    assert any("cardiac component NOT removed" in line for line in recording.log)


def test_with_heart_rate_the_mdn_filter_is_used():
    config = PreprocessingConfig(heart_rate_bpm=80)
    recording = preprocess(synthetic_raw()[0], config)
    assert any("MDN cardiac filter" in line for line in recording.log)
    assert recording.rates["heart_rate_hz"] == pytest.approx(80 / 60)


def test_an_implausible_heart_rate_is_used_with_a_warning():
    recording = preprocess(synthetic_raw()[0], PreprocessingConfig(heart_rate_bpm=250))
    assert any(line.startswith("WARNING: Heart rate 250") for line in recording.log)


def test_a_manual_exclusion_marks_the_breaths_it_touches():
    recording = preprocess(
        synthetic_raw()[0], exclusions=[Segment(30, 40, label="cough")]
    )
    breaths = recording.breaths
    excluded = breaths[breaths.excluded]
    assert len(excluded) >= 3 and (excluded.exclusion_label == "cough").all()
    assert not excluded.is_valid.any()
    assert (excluded.start_time <= 40.5).all() and (excluded.end_time >= 29.5).all()


def test_missing_frames_and_timing_errors_are_filled_and_excluded():
    raw, _ = synthetic_raw()
    raw.frames["pixels"][600:605, 10, 10] = np.nan
    raw.frames["timing_error"][1200] = 1
    recording = preprocess(raw)
    labels = {segment.label for segment in recording.exclusions}
    assert labels == {"missing_data", "timing_error"}
    assert all(segment.source == "device" for segment in recording.exclusions)
    assert np.isfinite(recording.pixels).all()
    assert recording.breaths.excluded.sum() >= 2


def test_the_parsed_data_is_not_modified():
    raw, _ = synthetic_raw()
    before = raw.frames["pixels"].copy()
    recording = preprocess(raw)
    np.testing.assert_array_equal(raw.frames["pixels"], before)
    assert not recording.pixels.flags.writeable


def test_a_recording_without_breaths_is_an_error():
    raw, _ = synthetic_raw()
    raw.frames["pixels"] = 1.0
    with pytest.raises(ValueError, match="No breaths"):
        preprocess(raw)


def test_data_without_frames_is_an_error():
    with pytest.raises(ValueError, match="frames"):
        preprocess(ReconstructedFrameData(fs=FS))


# --- real recordings ---


@pytest.mark.slow  # about 25 s
@pytest.mark.skipif(
    not (TEST_FILES / "patient02.bin").exists(), reason="real .bin files not available"
)
def test_real_recordings_give_the_breaths_of_the_evidence_notebook():
    for name, (n_breaths, ti, amplitude, duration) in NOTEBOOK_COUNTS.items():
        recording = preprocess(load_data(Path(TEST_FILES / name)))
        breaths = recording.breaths
        assert len(breaths) == n_breaths, name
        assert breaths.flag_suspect_ti_ratio.sum() == ti, name
        assert (~breaths.is_valid).sum() <= ti + amplitude + duration, name
        assert 300 < recording.lung_mask.sum() < 600, name
