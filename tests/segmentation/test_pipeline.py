"""Tests for segment_peep_trial()."""

import numpy as np
import pytest

from fasteit.config import SegmentationConfig
from fasteit.models.eit_recording import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.preprocessing.auxiliary import CMH2O_PER_MBAR
from fasteit.segmentation.pipeline import segment_peep_trial

from ..preprocessing.test_breath_pressures import pcv_pressure, vcv_pressure
from ..preprocessing.test_breath_quality import TEST_FILES
from ..preprocessing.test_pipeline import FS, synthetic_raw
from ..preprocessing.test_tidal_volume import vcv_flow, volume_curve

PEEPS = [12] * 12 + [10] * 12 + [8] * 12 + [6] * 14  # 50 breaths of 4 s


def synthetic_trial(aux=None, seconds=200, context=None):
    """A preprocessed recording at 15 breaths/min with a PEEP trial in its pressure."""
    raw, _ = synthetic_raw(seconds=seconds)
    if aux is None:
        paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0)
        aux = {"airway_pressure_pod": paw.astype(np.float32)}
    raw.aux_signals = aux
    return preprocess(raw, context=context)


@pytest.fixture(scope="module")
def recording():
    return synthetic_trial()


def test_steps_and_pressures_found_from_the_pressure(recording):
    trial = segment_peep_trial(recording)
    assert trial.source == "pressure"
    assert trial.recording.pressure_channel == "airway_pressure_pod"
    np.testing.assert_allclose(trial.steps.peep, [12, 10, 8, 6], atol=0.1)
    driving = trial.analysed_breaths.driving_pressure
    np.testing.assert_allclose(driving, 12, atol=0.2)


def test_step_table_has_the_driving_pressure(recording):
    trial = segment_peep_trial(recording)
    np.testing.assert_allclose(trial.steps.driving_pressure, 12, atol=0.2)
    assert trial.steps.enough_driving_pressure.all()
    assert (trial.steps.driving_pressure_source == "breaths").all()
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
    assert trial.recording.pressure_channel is None
    assert trial.breaths.peep.isna().all()
    assert any("No airway pressure" in line for line in trial.log)


def test_ventilator_values_are_used_without_a_waveform():
    n = int(200 * FS)
    peep = np.repeat([12.0, 10.0], [n // 2, n - n // 2]).astype(np.float32)
    recording = synthetic_trial(aux={"peep": peep, "plateau_pressure": peep + 15})
    trial = segment_peep_trial(recording)
    assert trial.recording.pressure_channel == "peep"
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

# Valid breaths after the first 15 s of steps 1-8 (notebook 06, section 5, in mbar).
NOTEBOOK_USABLE = {
    "A_park_bench": [5, 4, 6, 6, 6, 5, 5, 19],
    "A_supine": [5, 7, 8, 7, 6, 9, 6, 16],
    "B_park_bench": [20, 18, 13, 12, 16, 10, 8, 18],
    "B_supine": [13, 8, 9, 8, 9, 15, 12, 32],
}
# In cmH2O (x 1.02) the breath of the change to step 1 of A supine is 1.02 from the
# level instead of 0.99 mbar, so it opens the step: 4 s earlier, one more breath.
CMH2O_EXTRA = {"A_supine": [1, 0, 0, 0, 0, 0, 0, 0]}


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
        # Protocol in mbar, measured in cmH2O; PEEP 0 reads about 1 above (notebook 06).
        np.testing.assert_allclose(
            trial.steps.peep,
            np.arange(16, -1, -2) * CMH2O_PER_MBAR,
            atol=1.2,
            err_msg=name,
        )
        usable = np.add(usable, CMH2O_EXTRA.get(name, 0))
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


def test_one_inspiratory_hold_per_step_gives_the_driving_pressure():
    # Volume control without pause; in each step one breath with a 1.5 s hold.
    pauses = [1.5 if k % 12 == 6 else 0.0 for k in range(len(PEEPS))]
    paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0, pause=pauses)
    trial = segment_peep_trial(
        synthetic_trial(aux={"airway_pressure_pod": paw.astype(np.float32)})
    )
    assert (trial.steps.driving_pressure_source == "hold").all()
    assert (
        trial.steps.n_driving_pressure == 1
    ).all() and trial.steps.enough_driving_pressure.all()
    np.testing.assert_allclose(trial.steps.driving_pressure, 12, atol=0.2)


def test_pressure_control_gives_the_driving_pressure():
    paw, _ = pcv_pressure(PEEPS, fs=FS, period=4.0)
    trial = segment_peep_trial(
        synthetic_trial(aux={"airway_pressure_pod": paw.astype(np.float32)})
    )
    np.testing.assert_allclose(trial.steps.peep, [12, 10, 8, 6], atol=0.1)
    np.testing.assert_allclose(trial.steps.driving_pressure, 15, atol=0.2)


def test_a_driving_pressure_can_be_typed_instead_of_the_plateau():
    steps = [
        Segment(
            20,
            60,
            category="peep_step",
            parameters={"peep": 10, "driving_pressure": 14},
        )
    ]
    trial = segment_peep_trial(synthetic_trial(aux={}), peep_steps=steps)
    assert trial.steps.driving_pressure.tolist() == [14.0]
    assert trial.steps.plateau.tolist() == [24.0]
    assert trial.steps.driving_pressure_source.tolist() == ["typed"]


# --- tidal volume of each step ---

FOUR_STEPS = [
    Segment(20 + 40 * k, 56 + 40 * k, category="peep_step", parameters={"peep": p})
    for k, p in enumerate([12, 10, 8, 6])
]


def test_without_tidal_volume_the_steps_have_none(recording):
    trial = segment_peep_trial(recording)
    assert trial.steps.tidal_volume.isna().all()
    assert trial.steps.tidal_volume_source.isna().all()
    assert any("No tidal volume for the steps" in line for line in trial.log)


def test_a_tidal_volume_typed_for_the_recording_goes_to_every_step():
    recording = synthetic_trial(context=RecordingContext(tidal_volume_ml=450))
    trial = segment_peep_trial(recording)
    assert trial.steps.tidal_volume.tolist() == [450.0] * 4
    assert (trial.steps.tidal_volume_source == "typed_recording").all()


def test_a_tidal_volume_typed_for_each_step_wins(recording):
    trial = segment_peep_trial(recording, tidal_volumes=[400, 420, None, 460])
    assert trial.steps.tidal_volume.iloc[[0, 1, 3]].tolist() == [400.0, 420.0, 460.0]
    source = trial.steps.tidal_volume_source
    assert (source.iloc[[0, 1, 3]] == "typed").all() and source.isna().iloc[2]


def test_a_tidal_volume_typed_with_a_step_placed_by_hand(recording):
    steps = [*FOUR_STEPS[:3], Segment(140, 176, category="peep_step",
             parameters={"peep": 6, "tidal_volume": 380})]  # fmt: skip
    trial = segment_peep_trial(recording, peep_steps=steps)
    assert trial.steps.tidal_volume.iloc[3] == 380.0
    assert trial.steps.tidal_volume.iloc[:3].isna().all()


def test_the_tidal_volume_of_the_breaths_when_none_is_typed():
    paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0)
    flow, _, _, _ = vcv_flow([0.45] * len(PEEPS), fs=FS, period=4.0)
    aux = {
        "airway_pressure_pod": paw.astype(np.float32),
        "volume": volume_curve(flow, FS).astype(np.float32),  # Timpel: L
    }
    trial = segment_peep_trial(synthetic_trial(aux=aux))
    np.testing.assert_allclose(trial.steps.tidal_volume, 450, atol=2)
    assert (trial.steps.tidal_volume_source == "breaths").all()


@pytest.mark.parametrize(
    "steps, volumes, message",
    [
        ((), [450, 450], "2 values for 4 PEEP steps"),
        ((), [450, 450, -1, 450], "positive"),
        (
            [Segment(20, 56, category="peep_step",
                     parameters={"peep": 12, "tidal_volume": 450})],
            [450],
            "one place",
        ),
        (
            [Segment(20, 56, category="peep_step",
                     parameters={"peep": 12, "tidal_volume": -450})],
            None,
            "positive",
        ),
    ],
)  # fmt: skip
def test_unusable_typed_tidal_volumes_are_rejected(recording, steps, volumes, message):
    with pytest.raises(ValueError, match=message):
        segment_peep_trial(recording, peep_steps=steps, tidal_volumes=volumes)


def test_a_tidal_volume_of_none_in_a_segment_is_not_typed(recording):
    steps = [Segment(20, 56, category="peep_step",
                     parameters={"peep": 12, "tidal_volume": None})]  # fmt: skip
    trial = segment_peep_trial(recording, peep_steps=steps)
    assert trial.steps.tidal_volume.isna().all()


def trial_with_volume(context=None, **kwargs):
    """The synthetic trial with a measured tidal volume of 450 mL in every breath."""
    paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0)
    flow, _, _, _ = vcv_flow([0.45] * len(PEEPS), fs=FS, period=4.0)
    aux = {
        "airway_pressure_pod": paw.astype(np.float32),
        "volume": volume_curve(flow, FS).astype(np.float32),
    }
    return segment_peep_trial(synthetic_trial(aux=aux, context=context), **kwargs)


def test_a_typed_tidal_volume_far_from_the_measured_one_gives_a_warning():
    trial = trial_with_volume(context=RecordingContext(tidal_volume_ml=600))
    assert (trial.steps.tidal_volume_source == "typed_recording").all()
    assert any("tidal volume typed 600 mL, measured 450 mL" in m for m in trial.log)
    close = trial_with_volume(context=RecordingContext(tidal_volume_ml=470))
    assert not any("tidal volume typed" in m for m in close.log)


def test_a_tidal_volume_typed_for_a_step_in_litres_gives_a_warning(recording):
    trial = segment_peep_trial(recording, tidal_volumes=[0.45, None, None, None])
    assert any("Step 0: typed tidal volume 0.45 mL" in m for m in trial.log)
