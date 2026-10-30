"""Tests for segment_peep_trial()."""

import numpy as np
import pytest

from fasteit.config import SegmentationConfig
from fasteit.models.eit_recording import Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation.pipeline import segment_peep_trial

from ..preprocessing.test_breath_quality import TEST_FILES
from ..preprocessing.test_pipeline import FS, synthetic_raw
from .test_breath_pressures import vcv_pressure

PEEPS = [12] * 12 + [10] * 12 + [8] * 12 + [6] * 14  # 50 breaths of 4 s


def synthetic_trial(aux=None, seconds=200):
    """A preprocessed recording at 15 breaths/min with a PEEP trial in its pressure."""
    raw, _ = synthetic_raw(seconds=seconds)
    if aux is None:
        paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0)
        aux = {"airway_pressure_pod": paw.astype(np.float32)}
    raw.aux_signals = aux
    return preprocess(raw)


@pytest.fixture(scope="module")
def recording():
    return synthetic_trial()


def test_steps_and_pressures_found_from_the_pressure(recording):
    trial = segment_peep_trial(recording)
    assert (
        trial.source == "pressure" and trial.pressure_channel == "airway_pressure_pod"
    )
    np.testing.assert_allclose(trial.steps.peep, [12, 10, 8, 6], atol=0.1)
    driving = trial.analysed_breaths.driving_pressure
    np.testing.assert_allclose(driving, 12, atol=0.2)


def test_step_table_has_the_driving_pressure_and_the_unit(recording):
    trial = segment_peep_trial(recording)
    np.testing.assert_allclose(trial.steps.driving_pressure, 12, atol=0.2)
    assert trial.steps.enough_driving_pressure.all()
    assert trial.pressure_unit == "cmH2O"  # the synthetic recording is a Timpel one
    assert np.isfinite(trial.breaths.driving_pressure.iloc[-1])  # last breath too


def test_breaths_analysed_follow_the_rule(recording):
    config = SegmentationConfig()
    trial = segment_peep_trial(recording, config)
    b = trial.breaths
    for _, step in trial.steps.iterrows():
        analysed = b[(b.peep_step == step.step) & b.analysed]
        assert 1 <= len(analysed) <= config.analysis_max_breaths
        assert (
            analysed.start_time >= step.start_time + config.analysis_skip_seconds
        ).all()
        assert analysed.is_valid.all()
    assert trial.steps.n_analysed.tolist() == [
        len(trial.analysed_breaths[trial.analysed_breaths.peep_step == k])
        for k in range(4)
    ]


def test_a_breath_at_an_intermediate_peep_is_not_analysed():
    peeps = [12] * 11 + [11.2] + [10] * 12 + [8] * 12 + [6] * 14  # breath 11 halfway
    paw, _ = vcv_pressure(peeps, fs=FS, period=4.0)
    trial = segment_peep_trial(
        synthetic_trial(aux={"airway_pressure_pod": paw.astype(np.float32)})
    )
    halfway = trial.breaths[(trial.breaths.peep - 11.2).abs() < 0.1]
    assert len(halfway) == 1 and halfway.peep_step.iloc[0] == 0
    assert not halfway.analysed.iloc[0]


def test_too_few_breaths_give_a_warning(recording):
    trial = segment_peep_trial(recording, SegmentationConfig(analysis_min_breaths=9))
    assert not trial.steps.enough.all()
    assert any("fewer than 9" in line for line in trial.log)


def test_steps_placed_by_hand_replace_the_automatic_ones(recording):
    steps = [
        Segment(20, 44, category="peep_step", parameters={"peep": 12}),
        Segment(150, 190, category="peep_step", parameters={"peep": 6}),
    ]
    trial = segment_peep_trial(recording, peep_steps=steps)
    assert trial.source == "manual" and trial.steps.peep.tolist() == [12.0, 6.0]
    analysed = trial.analysed_breaths
    assert (analysed.start_time >= 20).all() and (analysed.end_time <= 190).all()
    assert analysed.is_valid.all() and (analysed.peep_step >= 0).all()
    assert not trial.breaths.analysed[trial.breaths.peep_step == -1].any()


def test_without_pressure_the_steps_must_be_placed_by_hand():
    recording = synthetic_trial(aux={})
    with pytest.raises(ValueError, match="place them by hand"):
        segment_peep_trial(recording)
    trial = segment_peep_trial(
        recording,
        peep_steps=[Segment(20, 60, category="peep_step", parameters={"peep": 10})],
    )
    assert trial.pressure_channel is None and trial.breaths.peep.isna().all()
    assert any("No airway pressure" in line for line in trial.log)


def test_ventilator_values_are_used_without_a_waveform():
    n = int(200 * FS)
    peep = np.repeat([12.0, 10.0], [n // 2, n - n // 2]).astype(np.float32)
    recording = synthetic_trial(aux={"peep": peep, "plateau_pressure": peep + 15})
    trial = segment_peep_trial(recording)
    assert trial.pressure_channel == "peep"
    np.testing.assert_allclose(trial.steps.peep, [12, 10], atol=0.01)
    np.testing.assert_allclose(trial.analysed_breaths.driving_pressure, 15, atol=0.01)


def test_without_end_inspiratory_pause_the_driving_pressure_is_missing():
    paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0, pause=0.0)
    trial = segment_peep_trial(
        synthetic_trial(aux={"airway_pressure_pod": paw.astype(np.float32)})
    )
    assert trial.breaths.driving_pressure.isna().all()
    assert any("No end-inspiratory plateau" in line for line in trial.log)


# --- real PEEP trials (notebook 06) ---

# Valid breaths after the first 15 s of steps 1-8 (notebook 06, section 5).
NOTEBOOK_USABLE = {
    "A_park_bench": [5, 4, 6, 6, 6, 5, 5, 19],
    "A_supine": [5, 7, 8, 7, 6, 9, 6, 16],
    "B_park_bench": [20, 18, 13, 12, 16, 10, 8, 18],
    "B_supine": [13, 8, 9, 8, 9, 15, 12, 32],
}


@pytest.mark.slow  # about 25 s
@pytest.mark.skipif(
    not all((TEST_FILES / f"PEEP_trial_{n}.bin").exists() for n in NOTEBOOK_USABLE),
    reason="real PEEP trials not available",
)
def test_real_peep_trials_give_the_steps_and_breaths_of_notebook_06():
    for name, usable in NOTEBOOK_USABLE.items():
        trial = segment_peep_trial(
            preprocess(load_data(TEST_FILES / f"PEEP_trial_{name}.bin"))
        )
        assert len(trial.steps) == 9, name
        np.testing.assert_allclose(
            trial.steps.peep, np.arange(16, -1, -2), atol=1.1, err_msg=name
        )
        assert trial.steps.n_analysed.iloc[1:].tolist() == [
            min(n, 10) for n in usable
        ], name
        assert trial.analysed_breaths.plateau.notna().mean() > 0.95, (
            name
        )  # 99-100 % in notebook 06


# --- a recording with more than one protocol ---

BASELINE_THEN_TRIAL = [12] * 10 + [16] * 10 + [14] * 10 + [12] * 10 + [10] * 10


def baseline_then_trial(exclusions=()):
    raw, _ = synthetic_raw(seconds=200)
    paw, _ = vcv_pressure(BASELINE_THEN_TRIAL, fs=FS, period=4.0)
    raw.aux_signals = {"airway_pressure_pod": paw.astype(np.float32)}
    return preprocess(raw, exclusions=exclusions)


TRIAL_PHASE = Segment(40, 200, category="phase", label="peep_trial")


def test_the_phase_limits_the_steps_to_the_peep_trial():
    recording = baseline_then_trial()
    whole = segment_peep_trial(recording)
    assert whole.steps.peep.round().tolist() == [
        12,
        16,
        14,
        12,
        10,
    ]  # baseline as a step
    trial = segment_peep_trial(recording, phase=TRIAL_PHASE)
    assert trial.steps.peep.round().tolist() == [16, 14, 12, 10]
    before = trial.breaths[trial.breaths.end_time <= 40]
    assert (before.peep_step == -1).all() and not before.analysed.any()


def test_a_maneuver_excluded_in_preprocess_is_never_analysed():
    recording = baseline_then_trial(exclusions=[Segment(170, 200, label="aop")])
    trial = segment_peep_trial(recording, phase=TRIAL_PHASE)
    excluded = trial.breaths[trial.breaths.excluded]
    assert len(excluded) > 0 and not excluded.analysed.any()


@pytest.mark.parametrize(
    "phase, message",
    [
        (Segment(40, 200, label="cough"), "category"),
        (Segment(40, 200, category="phase", label="tidal"), "peep_trial"),
        (Segment(500, 600, category="phase", label="peep_trial"), "No breath inside"),
    ],
)
def test_unusable_phases_are_rejected(phase, message):
    with pytest.raises(ValueError, match=message):
        segment_peep_trial(baseline_then_trial(), phase=phase)


def test_an_excluded_maneuver_at_another_peep_does_not_open_a_step():
    peeps = [12] * 12 + [10] * 6 + [16] * 4 + [10] * 2 + [8] * 12 + [6] * 14
    paw, _ = vcv_pressure(peeps, fs=FS, period=4.0)
    raw, _ = synthetic_raw(seconds=200)
    raw.aux_signals = {"airway_pressure_pod": paw.astype(np.float32)}
    maneuver = Segment(72, 88, label="aop")  # the 4 breaths at 16
    trial = segment_peep_trial(preprocess(raw, exclusions=[maneuver]))
    assert trial.steps.peep.round().tolist() == [12, 10, 8, 6]


def test_a_pressure_missing_in_most_breaths_gives_a_warning():
    paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0)
    paw = paw.astype(np.float32)
    paw[: int(0.7 * len(paw))] = np.nan  # e.g. the PressurePod connected late
    trial = segment_peep_trial(synthetic_trial(aux={"airway_pressure_pod": paw}))
    assert any("PEEP missing in most breaths" in line for line in trial.log)


def test_a_typed_peep_far_from_the_measured_one_gives_a_warning(recording):
    steps = [
        Segment(20, 44, category="peep_step", parameters={"peep": 16})
    ]  # 12 measured
    trial = segment_peep_trial(recording, peep_steps=steps)
    assert any("PEEP typed 16, measured 12.0" in line for line in trial.log)


def test_a_typed_plateau_gives_the_driving_pressure_without_pressure():
    steps = [
        Segment(20, 60, category="peep_step", parameters={"peep": 10, "plateau": 25})
    ]
    trial = segment_peep_trial(synthetic_trial(aux={}), peep_steps=steps)
    assert trial.steps.driving_pressure.tolist() == [15.0]
    assert trial.steps.enough_driving_pressure.all()


def test_without_any_plateau_the_log_says_compliance_is_not_available():
    steps = [Segment(20, 60, category="peep_step", parameters={"peep": 10})]
    trial = segment_peep_trial(synthetic_trial(aux={}), peep_steps=steps)
    assert any("compliance cannot be computed" in line for line in trial.log)
