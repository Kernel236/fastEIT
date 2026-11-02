"""Tests for preprocess()."""

from pathlib import Path

import numpy as np
import pytest
from scipy import signal as scipy_signal

from fasteit.config import PreprocessingConfig
from fasteit.models.eit_recording import RecordingContext, Segment
from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.loader import load_data
from fasteit.preprocessing.auxiliary import CMH2O_PER_MBAR
from fasteit.preprocessing.pipeline import preprocess, recording_time

from .test_breath_pressures import vcv_pressure
from .test_breath_quality import NOTEBOOK_COUNTS, TEST_FILES
from .test_tidal_volume import vcv_flow, volume_curve

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


def test_the_cardiac_filter_applied_is_recorded():
    without = preprocess(synthetic_raw()[0])
    assert without.cardiac_filter == "moving_average"
    assert any("cardiac component NOT removed" in line for line in without.log)
    with_heart = preprocess(synthetic_raw()[0], PreprocessingConfig(heart_rate_bpm=80))
    assert with_heart.cardiac_filter == "mdn"
    assert any("MDN cardiac filter" in line for line in with_heart.log)
    assert with_heart.rates["heart_rate_hz"] == pytest.approx(80 / 60)
    # A heart rate below the respiratory rate cannot be filtered: fallback, logged.
    slow_heart = preprocess(synthetic_raw()[0], PreprocessingConfig(heart_rate_bpm=10))
    assert slow_heart.cardiac_filter == "moving_average"
    assert any("NOT removed" in line for line in slow_heart.log)


def test_mdn_removes_a_cardiac_oscillation_from_the_pixels():
    raw, lungs = synthetic_raw()
    t = np.arange(len(raw.frames)) / FS
    heart = 0.05 * np.sin(2 * np.pi * (80 / 60) * t)
    raw.frames["pixels"] += (heart[:, None, None] * lungs).astype(np.float32)

    def cardiac_power(recording):
        frequencies, power = scipy_signal.welch(
            recording.global_signal, FS, nperseg=600
        )
        return power[np.abs(frequencies - 80 / 60) < 0.1].sum()

    unfiltered = preprocess(raw)
    filtered = preprocess(raw, PreprocessingConfig(heart_rate_bpm=80))
    assert cardiac_power(filtered) < cardiac_power(unfiltered) / 100


def test_draeger_no_data_values_become_excluded_frames():
    raw, _ = synthetic_raw(vendor="draeger")
    raw.frames["pixels"][600:603, 10, 10] = -1000.0
    raw.frames["pixels"][900, 12, 12] = np.array([0xFF7FC99E], np.uint32).view(
        np.float32
    )[0]
    recording = preprocess(raw)
    assert [s.label for s in recording.exclusions] == ["missing_data", "missing_data"]
    assert np.isfinite(recording.pixels).all()
    assert raw.frames["pixels"][600, 10, 10] == -1000.0  # parsed data untouched


def test_a_gap_in_the_frame_times_is_excluded():
    raw, _ = synthetic_raw()
    raw.frames["ts"][1000:] += 0.2  # 4 frames dropped at 50 s
    recording = preprocess(raw)
    assert [s.label for s in recording.exclusions] == ["time_gap"]
    assert recording.breaths.excluded.sum() >= 1


def test_time_going_back_is_an_error():
    raw, _ = synthetic_raw()
    raw.frames["ts"][1000:] -= 1.0
    with pytest.raises(ValueError, match="increase"):
        preprocess(raw)


def test_breath_table_and_regions_have_stable_types():
    recording = preprocess(
        synthetic_raw()[0], exclusions=[Segment(30, 40, label="cough")]
    )
    labels = recording.breaths.exclusion_label
    assert labels.dtype == "string" and labels.isna().any()
    roi = recording.roi_signals
    np.testing.assert_allclose(roi.ventral + roi.dorsal, recording.global_signal)
    assert recording.image_mask.sum() > recording.lung_mask.sum()


def test_an_empty_region_has_a_nan_signal_and_a_warning():
    raw, lungs = synthetic_raw()
    raw.frames["pixels"][:, :, 16:] = 0.0  # one lung only (patient right)
    recording = preprocess(raw)
    assert recording.roi_signals.left.isna().all()
    assert any("Region left is empty" in line for line in recording.log)


def test_auxiliary_signals_are_cleaned_and_aligned_with_the_frames():
    raw, _ = synthetic_raw()
    pressure = np.full(len(raw.frames), 5.0, dtype=np.float32)
    pressure[100] = np.nan
    raw.aux_signals = {"airway_pressure": pressure}
    recording = preprocess(raw)
    assert len(recording.aux) == len(recording.time)
    assert recording.aux.airway_pressure.isna().sum() == 1
    assert any(
        "Auxiliary signals with data: airway_pressure" in m for m in recording.log
    )


def test_the_pressures_of_each_breath_are_read_on_the_airway_pressure():
    raw, _ = synthetic_raw()
    paw, _ = vcv_pressure([10] * 23, fs=FS, period=4.0)  # 15 breaths/min
    raw.aux_signals = {"airway_pressure": paw[: len(raw.frames)].astype(np.float32)}
    recording = preprocess(raw)
    breaths = recording.breaths
    assert recording.pressure_channel == "airway_pressure"
    np.testing.assert_allclose(breaths.peep, 10, atol=0.1)
    np.testing.assert_allclose(breaths.driving_pressure.iloc[1:-1], 12, atol=0.2)
    assert any("PEEP and plateau of each breath read on" in m for m in recording.log)


def test_draeger_pressures_of_each_breath_are_in_cmh2o():
    raw, _ = synthetic_raw(vendor="draeger")
    paw, _ = vcv_pressure([10] * 23, fs=FS, period=4.0)  # in mbar, as recorded
    raw.aux_signals = {"airway_pressure_pod": paw[: len(raw.frames)].astype(np.float32)}
    breaths = preprocess(raw).breaths
    np.testing.assert_allclose(breaths.peep, 10.2, atol=0.1)
    np.testing.assert_allclose(breaths.driving_pressure.iloc[1:-1], 12.24, atol=0.2)


def test_draeger_ventilator_pressures_without_a_waveform_are_in_cmh2o():
    raw, _ = synthetic_raw(vendor="draeger")
    n = len(raw.frames)
    raw.aux_signals = {  # Medibus values, in mbar
        "peep": np.full(n, 10, dtype=np.float32),
        "plateau_pressure": np.full(n, 25, dtype=np.float32),
    }
    recording = preprocess(raw)
    assert recording.pressure_channel == "peep"
    np.testing.assert_allclose(recording.breaths.peep, 10 * CMH2O_PER_MBAR, rtol=1e-6)
    np.testing.assert_allclose(
        recording.breaths.driving_pressure, 15 * CMH2O_PER_MBAR, rtol=1e-6
    )


def test_the_pressurepod_comes_before_the_ventilator_pressure():
    raw, _ = synthetic_raw()
    paw, _ = vcv_pressure([10] * 23, fs=FS, period=4.0)
    paw = paw[: len(raw.frames)].astype(np.float32)
    raw.aux_signals = {"airway_pressure": paw + 5, "airway_pressure_pod": paw}
    recording = preprocess(raw)
    assert recording.pressure_channel == "airway_pressure_pod"
    np.testing.assert_allclose(recording.breaths.peep, 10, atol=0.1)
    # A PressurePod without data is dropped: the ventilator pressure is used.
    raw.aux_signals = {
        "airway_pressure": paw + 5,
        "airway_pressure_pod": np.full_like(paw, np.nan),
    }
    assert preprocess(raw).pressure_channel == "airway_pressure"


def test_without_pressure_the_pressures_of_the_breaths_are_missing():
    recording = preprocess(synthetic_raw()[0])
    assert recording.pressure_channel is None
    assert recording.breaths[["peep", "plateau", "driving_pressure"]].isna().all().all()
    assert not any(line.startswith("WARNING: No airway") for line in recording.log)


def ventilator_curves(n_frames):
    """Flow (L/s) and volume (L) of a breath of 450 mL every 4 s, as in synthetic_raw."""
    flow, _, _, _ = vcv_flow([0.45] * 23, fs=FS, period=4.0)
    flow = flow[:n_frames]
    return flow.astype(np.float32), volume_curve(flow, FS).astype(np.float32)


def middle(breaths):
    return breaths.tidal_volume.iloc[1:-1]


def test_tidal_volume_of_each_breath_from_the_timpel_volume_curve():
    raw, _ = synthetic_raw()
    flow, volume = ventilator_curves(len(raw.frames))
    raw.aux_signals = {"flow": flow, "volume": volume}  # Timpel: L/s and L
    recording = preprocess(raw)
    assert recording.tidal_volume_channel == "volume"  # the volume curve first
    np.testing.assert_allclose(middle(recording.breaths), 450, atol=2)


def test_tidal_volume_of_each_breath_from_the_draeger_flow_curve():
    raw, _ = synthetic_raw(vendor="draeger")
    flow, _ = ventilator_curves(len(raw.frames))
    raw.aux_signals = {"flow": flow * 60}  # Draeger Medibus: L/min
    recording = preprocess(raw)
    assert recording.tidal_volume_channel == "flow"
    np.testing.assert_allclose(middle(recording.breaths), 450, atol=2)


def test_tidal_volume_computed_by_the_ventilator_without_curves():
    raw, _ = synthetic_raw(vendor="draeger")
    raw.aux_signals = {"tidal_volume": np.full(len(raw.frames), 480, dtype=np.float32)}
    recording = preprocess(raw)
    assert recording.tidal_volume_channel == "tidal_volume"
    np.testing.assert_allclose(recording.breaths.tidal_volume, 480)


def test_the_inspiratory_values_of_the_ventilator_come_first():
    raw, _ = synthetic_raw(vendor="draeger")
    n = len(raw.frames)
    raw.aux_signals = {
        "tidal_volume": np.full(n, 430, dtype=np.float32),  # phase not known
        "mandatory_inspiratory_tidal_volume": np.full(n, 450, dtype=np.float32),
    }
    recording = preprocess(raw)
    assert recording.tidal_volume_channel == "mandatory_inspiratory_tidal_volume"
    np.testing.assert_allclose(recording.breaths.tidal_volume, 450)


def test_a_flow_negative_during_inspiration_gives_a_warning():
    raw, _ = synthetic_raw()
    flow, _ = ventilator_curves(len(raw.frames))
    raw.aux_signals = {"flow": -flow}  # inspiration recorded as negative
    recording = preprocess(raw)
    assert any("flow does not rise" in line for line in recording.log)


def test_a_typed_tidal_volume_in_litres_gives_a_warning():
    recording = preprocess(
        synthetic_raw()[0], context=RecordingContext(tidal_volume_ml=0.45)
    )
    assert any("is it in mL?" in line for line in recording.log)
    recording = preprocess(
        synthetic_raw()[0], context=RecordingContext(tidal_volume_ml=450)
    )
    assert not any("is it in mL?" in line for line in recording.log)


def test_without_volume_or_flow_the_tidal_volume_is_missing():
    recording = preprocess(synthetic_raw()[0])
    assert recording.tidal_volume_channel is None
    assert recording.breaths.tidal_volume.isna().all()
    assert any("no tidal volume of the breaths" in line for line in recording.log)


def test_vendors_other_than_draeger_get_a_warning():
    recording = preprocess(synthetic_raw(vendor="timpel")[0])
    assert any("verified on Draeger" in line for line in recording.log)


# --- real recordings ---


@pytest.mark.slow  # about 25 s
@pytest.mark.skipif(
    not all((TEST_FILES / name).exists() for name in NOTEBOOK_COUNTS),
    reason="real .bin files not available",
)
def test_real_recordings_give_the_breaths_of_the_evidence_notebook():
    for name, (n_breaths, ti, amplitude, duration) in NOTEBOOK_COUNTS.items():
        recording = preprocess(load_data(Path(TEST_FILES / name)))
        breaths = recording.breaths
        assert len(breaths) == n_breaths, name
        assert breaths.flag_suspect_ti_ratio.sum() == ti, name
        flagged = ~breaths.is_valid & ~breaths.excluded
        assert flagged.sum() <= ti + amplitude + duration, name
        assert 300 < recording.lung_mask.sum() < 600, name
        if name.startswith("patient01"):  # PressurePod connected, no ventilator
            assert "esophageal_pressure_pod" in recording.aux, name
            assert "peep" not in recording.aux, name
            assert recording.pressure_channel == "airway_pressure_pod", name
            # 85-100 %: a single missing frame of the PressurePod loses the PEEP of
            # the breath around it (up to 12 such frames per recording).
            assert recording.breaths.peep.notna().mean() > 0.8, name
            assert recording.aux.min_max_flag.isin([-1, 0, 1]).all(), name
