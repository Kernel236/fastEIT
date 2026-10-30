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
    assert trial.steps.analysed.tolist() == [
        len(trial.analysed_breaths[trial.analysed_breaths.peep_step == k])
        for k in range(4)
    ]


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
    "B_park_bench": [20, 19, 13, 12, 16, 10, 8, 20],
    "B_supine": [14, 8, 11, 9, 9, 15, 12, 35],
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
        assert trial.steps.analysed.iloc[1:].tolist() == [min(n, 10) for n in usable], (
            name
        )
        assert trial.analysed_breaths.plateau.notna().mean() > 0.95, (
            name
        )  # 99-100 % in notebook 06
