"""Tests for tidal_features() and peep_trial_features()."""

import numpy as np
import pandas as pd
import pytest

from fasteit.config import AnalysisConfig, SegmentationConfig
from fasteit.features import peep_trial_features, tidal_features
from fasteit.features.pipeline import _add_changes
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

from ..preprocessing.test_breath_pressures import vcv_pressure
from ..preprocessing.test_breath_quality import TEST_FILES
from ..preprocessing.test_pipeline import FS, synthetic_raw
from ..preprocessing.test_tidal_volume import vcv_flow, volume_curve
from ..segmentation.test_pipeline import FOUR_STEPS, PEEPS, synthetic_trial, with_typed


def tidal_recording(context=None, volume=False):
    """90 s at 15 breaths/min, PEEP 10 and driving pressure 12 (synthetic_raw)."""
    raw, _ = synthetic_raw()
    n = len(raw.frames)
    paw, _ = vcv_pressure([10] * 23, fs=FS, period=4.0)
    raw.aux_signals = {"airway_pressure": paw[:n].astype(np.float32)}
    if volume:
        flow, _, _, _ = vcv_flow([0.45] * 23, fs=FS, period=4.0)
        raw.aux_signals["volume"] = volume_curve(flow[:n], FS).astype(np.float32)
    return preprocess(raw, context=context)


@pytest.fixture(scope="module")
def tidal():
    return tidal_features(tidal_recording(RecordingContext(tidal_volume_ml=450)))


def middle(values):
    return values.iloc[1:-1]  # the first and last breaths have no plateau


def test_breath_indices_and_compliance(tidal):
    b, measured = tidal.breaths, tidal.recording.breaths
    assert len(b) == len(measured)
    assert (b.tidal_volume_source == "typed_recording").all()
    np.testing.assert_allclose(middle(measured.driving_pressure), 12, atol=0.2)
    np.testing.assert_allclose(middle(b.compliance_ml), 450 / 12, rtol=0.02)
    np.testing.assert_allclose(b.compliance, b.tiv / measured.driving_pressure)
    np.testing.assert_allclose(b.respiratory_rate, 15, atol=0.5)
    np.testing.assert_allclose(b.cov_rl, 0.5, atol=0.02)  # two symmetric lungs


def test_regions_share_the_tidal_volume(tidal):
    r = tidal.regions
    halves = r[r.region.isin(["ventral", "dorsal"])].groupby("breath_id")
    np.testing.assert_allclose(halves.share.sum(), 1, atol=1e-9)
    np.testing.assert_allclose(halves.tiv_ml.sum(), 450, atol=1e-6)
    right = r[r.region == "right"]
    np.testing.assert_allclose(right.share, 0.5, atol=0.02)
    np.testing.assert_allclose(
        right.compliance_ml,
        right.share * 450 / tidal.recording.breaths.driving_pressure.values,
    )


def test_the_summary_is_over_the_valid_breaths(tidal):
    s = tidal.summary
    row = s[(s.region == "global") & (s.feature == "compliance_ml")].iloc[0]
    assert row["median"] == pytest.approx(450 / 12, rel=0.02)
    assert row.q1 <= row["median"] <= row.q3
    tiv = s[(s.region == "global") & (s.feature == "tiv")].iloc[0]
    assert tiv.n_breaths == tidal.recording.breaths.is_valid.sum()
    peep = s[(s.region == "global") & (s.feature == "peep")].iloc[0]
    assert peep["median"] == pytest.approx(10, abs=0.1)  # pressures of layer 2 too
    assert set(s.region) == {"global", *tidal.recording.roi_masks}


def test_without_pressure_and_volume_the_values_in_ml_and_compliance_are_missing():
    features = tidal_features(preprocess(synthetic_raw()[0]))
    b = features.breaths
    assert b.compliance.isna().all() and b.compliance_ml.isna().all()
    assert b.tidal_volume_source.isna().all()
    assert features.regions.tiv_ml.isna().all()
    assert any("No tidal volume" in line for line in features.log)
    assert np.isfinite(b.gi).all() and np.isfinite(b.tiv).all()  # EIT indices are there


def test_the_measured_tidal_volume_is_used_without_a_typed_one():
    features = tidal_features(tidal_recording(volume=True))
    b = features.breaths
    assert (middle(b.tidal_volume_source) == "breaths").all()
    np.testing.assert_allclose(middle(b.tidal_volume_used), 450, atol=2)


def test_the_tidal_phase_limits_the_summary():
    recording = tidal_recording()
    features = tidal_features(
        recording, phase=Segment(40, 90, category="phase", label="tidal")
    )
    inside = recording.breaths[features.breaths.in_phase.to_numpy()]
    assert (inside.start_time >= 40).all()
    tiv = features.summary.query("region == 'global' and feature == 'tiv'").iloc[0]
    assert tiv.n_breaths == inside.is_valid.sum()
    with pytest.raises(ValueError, match="tidal"):
        tidal_features(
            recording, phase=Segment(40, 90, category="phase", label="peep_trial")
        )


def test_a_phase_without_valid_breaths_keeps_every_region_in_the_summary():
    recording = preprocess(
        tidal_recording().raw, exclusions=[Segment(40, 90, label="cough")]
    )
    phase = Segment(45, 90, category="phase", label="tidal")
    features = tidal_features(recording, phase=phase)
    assert (features.summary.n_breaths == 0).all()
    assert set(features.summary.region) == {"global", *recording.roi_masks}
    assert any("No valid breath" in line for line in features.log)
    with pytest.raises(ValueError, match="No breath inside"):
        tidal_features(
            recording, phase=Segment(200, 300, category="phase", label="tidal")
        )


# --- PEEP trials ---


def test_changes_between_steps_with_a_known_answer():
    steps = pd.DataFrame({
        "peep": [16.0, 14.0, 12.0], "eeli": [100.0, 80.0, 50.0], "tiv": [20.0, 25.0, 20.0],
        "compliance_pct_of_best": [90.0, 100.0, 80.0],
        "compliance_ml_pct_of_best": [90.0, 100.0, 80.0],
    })  # fmt: skip
    c = _add_changes(steps, tidal_volume=np.array([500.0, 500.0, 400.0]))
    # Converted with the TIV and tidal volume of the step at the lower PEEP (each step).
    np.testing.assert_allclose(c.d_eeli_tiv_vs_previous, [np.nan, -20 / 25, -30 / 20])
    np.testing.assert_allclose(c.d_eeli_tiv_vs_first, [0.0, -20 / 25, -50 / 20])
    np.testing.assert_allclose(c.d_eeli_ml_vs_previous, [np.nan, -400.0, -600.0])
    np.testing.assert_allclose(c.d_eeli_ml_vs_first, [0.0, -400.0, -1000.0])
    np.testing.assert_allclose(
        c.d_compliance_pct_of_best_vs_previous, [np.nan, 10, -20]
    )
    np.testing.assert_allclose(c.d_compliance_pct_of_best_vs_first, [0, 10, -10])


def test_in_an_incremental_trial_the_lower_step_is_the_reference():
    steps = pd.DataFrame({
        "peep": [6.0, 8.0], "eeli": [50.0, 80.0], "tiv": [20.0, 25.0],
        "compliance_pct_of_best": [90.0, 100.0], "compliance_ml_pct_of_best": [90.0, 100.0],
    })  # fmt: skip
    c = _add_changes(steps, tidal_volume=np.array([400.0, 500.0]))
    assert c.d_eeli_tiv_vs_previous[1] == pytest.approx(30 / 20)
    assert c.d_eeli_ml_vs_previous[1] == pytest.approx(30 / 20 * 400)


@pytest.fixture(scope="module")
def trial_features():
    recording = synthetic_trial(context=RecordingContext(tidal_volume_ml=450))
    return peep_trial_features(segment_peep_trial(recording))


def test_one_row_per_step_with_compliance_and_changes(trial_features):
    s, layer_2b = trial_features.steps, trial_features.trial.steps
    np.testing.assert_allclose(s.peep, [12, 10, 8, 6], atol=0.1)
    np.testing.assert_allclose(s.compliance_ml, 450 / 12, rtol=0.02)
    np.testing.assert_allclose(s.compliance, s.tiv / layer_2b.driving_pressure)
    assert s.compliance_pct_of_best.max() == pytest.approx(100)
    # The same ventilation at every step: no collapse or overdistension to speak of.
    assert s.collapse.max() < 2 and s.overdistension.max() < 2
    assert np.isnan(s.d_eeli_tiv_vs_previous[0]) and s.d_eeli_ml_vs_first[0] == 0
    np.testing.assert_allclose(
        s.d_eeli_ml_vs_first, s.d_eeli_tiv_vs_first * layer_2b.tidal_volume[0]
    )


def test_step_regions_and_summary(trial_features):
    r = trial_features.step_regions
    halves = r[r.region.isin(["ventral", "dorsal"])].groupby("step")
    np.testing.assert_allclose(halves.share.sum(), 1, atol=0.02)
    np.testing.assert_allclose(halves.tiv_ml.sum(), 450, rtol=0.02)
    summary = trial_features.summary
    assert set(summary.columns) == {"region", "feature", "value"}
    # No R/I levels in the config: no R/I, and the log says why.
    assert "ri" not in set(summary.feature)
    assert any("No R/I" in line for line in trial_features.log)


def test_layer_3_tables_do_not_copy_what_layers_2_and_2b_measured(trial_features):
    trial = trial_features.trial
    breaths_before = set(trial.recording.breaths.columns) | set(trial.breaths.columns)
    assert set(trial_features.breaths.columns) & breaths_before == {"breath_id"}
    assert set(trial_features.steps.columns) & set(trial.steps.columns) == {
        "step",
        "peep",
    }


def test_a_step_without_analysed_breaths_has_its_rows_and_a_warning():
    raw, _ = synthetic_raw(seconds=200)
    paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0)
    raw.aux_signals = {"airway_pressure_pod": paw.astype(np.float32)}
    recording = preprocess(raw, exclusions=[Segment(138, 180, label="cough")])
    features = peep_trial_features(segment_peep_trial(recording, peep_steps=FOUR_STEPS))
    s, layer_2b = features.steps, features.trial.steps
    assert layer_2b.n_analysed.iloc[3] == 0 and not layer_2b.enough.iloc[3]
    assert np.isnan(s.compliance.iloc[3]) and np.isnan(s.collapse.iloc[3])
    assert np.isfinite(s.collapse.iloc[:3]).all()
    last = features.step_regions[features.step_regions.step == 3]
    assert set(last.region) == set(recording.roi_masks) and last.tiv.isna().all()
    assert any("have no compliance" in line for line in features.log)
    assert any("fewer than 5 analysed breaths" in line for line in features.log)


def test_a_first_step_without_breaths_gives_a_warning():
    raw, _ = synthetic_raw(seconds=200)
    paw, _ = vcv_pressure(PEEPS, fs=FS, period=4.0)
    raw.aux_signals = {"airway_pressure_pod": paw.astype(np.float32)}
    recording = preprocess(raw, exclusions=[Segment(15, 60, label="cough")])
    features = peep_trial_features(segment_peep_trial(recording, peep_steps=FOUR_STEPS))
    assert features.steps.d_eeli_tiv_vs_first.isna().all()
    assert any("The first step has no EELI" in line for line in features.log)


def test_a_tidal_volume_typed_for_a_step_goes_to_its_breaths():
    trial = segment_peep_trial(synthetic_trial(), tidal_volumes=[400, 420, 440, 460])
    features = peep_trial_features(trial)
    b, step = features.breaths, trial.breaths.peep_step
    in_step_1 = b[step == 1]
    assert (in_step_1.tidal_volume_used == 420).all()
    assert (in_step_1.tidal_volume_source == "typed").all()
    assert b[step == -1].tidal_volume_used.isna().all()
    np.testing.assert_allclose(
        features.steps.compliance_ml,
        np.array([400, 420, 440, 460]) / trial.steps.driving_pressure,
    )


def test_a_driving_pressure_typed_for_a_step_goes_to_its_breaths():
    # Plateau 25 typed for step 1 (PEEP 10): driving pressure 15, the curve reads 12.
    recording = synthetic_trial(context=RecordingContext(tidal_volume_ml=450))
    trial = segment_peep_trial(recording, peep_steps=with_typed({1: {"plateau": 25}}))
    features = peep_trial_features(trial)
    b, step = features.breaths, trial.breaths.peep_step
    in_step_1, in_step_2 = b[step == 1], b[step == 2]
    assert (in_step_1.driving_pressure_used == 15).all()
    assert (in_step_1.driving_pressure_source == "typed").all()
    np.testing.assert_allclose(in_step_1.compliance_ml, 450 / 15)
    measured = trial.recording.breaths.driving_pressure[step == 2]
    np.testing.assert_allclose(in_step_2.driving_pressure_used, measured)
    assert (in_step_2.driving_pressure_source == "breaths").all()
    # The breaths and the step give the same compliance, globally and by region.
    used_1 = trial.breaths.analysed & (step == 1)
    assert features.steps.compliance_ml[1] == pytest.approx(450 / 15)
    assert features.steps.compliance[1] == pytest.approx(b.compliance[used_1].median())
    regions = features.regions[features.regions.region == "ventral"]
    np.testing.assert_allclose(
        regions.compliance.to_numpy()[used_1], regions.tiv.to_numpy()[used_1] / 15
    )


# --- real PEEP trials (notebook 07, sections 3-4) ---

SET_VT = {"A": 550, "B": 400}
PROTOCOL = SegmentationConfig(peep_levels=(16, 14, 12, 10, 8, 6, 4, 2, 0))
RI_14_4 = AnalysisConfig(ri_peep_high=14, ri_peep_low=4)
# Crossing PEEP, best PEEP of Vt / driving (both on the levels of the protocol), EELI
# change in mL at the end, R/I between the steps at 14 and 4.
NOTEBOOK_07 = {
    "A_park_bench": (4.6, 6, -1928, 0.74),
    "A_supine": (6.5, 4, -2484, 0.93),
    "B_park_bench": (9.6, 8, -1508, 1.47),
    "B_supine": (12.2, 12, -1141, 1.32),
}


def value(features, feature, region="global"):
    """A value of the summary of a PEEP trial."""
    summary = features.summary
    rows = summary[(summary.region == region) & (summary.feature == feature)]
    return rows.value.iloc[0]


@pytest.mark.slow  # about 25 s
@pytest.mark.skipif(
    not all((TEST_FILES / f"PEEP_trial_{n}.bin").exists() for n in NOTEBOOK_07),
    reason="real PEEP trials not available",
)
def test_real_peep_trials_give_the_indices_of_notebook_07():
    for name, (crossing, best_ml, eeli_ml, ri) in NOTEBOOK_07.items():
        context = RecordingContext(
            protocol="peep_trial", tidal_volume_ml=SET_VT[name[0]]
        )
        recording = preprocess(
            load_data(TEST_FILES / f"PEEP_trial_{name}.bin"), context=context
        )
        features = peep_trial_features(segment_peep_trial(recording, PROTOCOL), RI_14_4)
        assert value(features, "crossing_peep") == pytest.approx(crossing, abs=0.05), (
            name
        )
        assert value(features, "best_peep_compliance_ml") == pytest.approx(
            best_ml, abs=0.05
        ), name
        assert value(features, "ri") == pytest.approx(ri, abs=0.01), name
        last = features.steps.d_eeli_ml_vs_first.iloc[-1]
        assert last == pytest.approx(eeli_ml, abs=1), name


RI_12_6 = AnalysisConfig(ri_peep_high=12, ri_peep_low=6)
LEVELS = SegmentationConfig(peep_levels=(12, 10, 8, 6))


def test_results_of_the_trial_on_the_protocol_levels():
    recording = synthetic_trial(context=RecordingContext(tidal_volume_ml=450))
    measured = peep_trial_features(segment_peep_trial(recording), RI_12_6)
    protocol = peep_trial_features(segment_peep_trial(recording, LEVELS), RI_12_6)
    assert value(protocol, "best_peep_compliance") in {12, 10, 8, 6}
    # No protocol levels: the PEEP of the steps (measured, or typed by hand).
    on_steps = (
        "on the PEEP of the steps (measured, or typed with the steps placed by hand)"
    )
    assert any(on_steps in m for m in measured.log)
    assert any(on_steps in m for m in measured.trial.log)
    assert any("PEEP levels of the protocol" in m for m in protocol.log)


def test_ri_only_between_the_two_levels_of_the_protocol_of_the_config():
    recording = synthetic_trial(context=RecordingContext(tidal_volume_ml=450))
    trial = segment_peep_trial(recording, LEVELS)
    # The steps are chosen on the levels of the protocol, never by nearness: 15 and 5
    # are not levels, and without the levels of the protocol no step is at a level.
    for config, steps in [
        (AnalysisConfig(ri_peep_high=15, ri_peep_low=5), trial),
        (RI_12_6, segment_peep_trial(recording)),
    ]:
        features = peep_trial_features(steps, config)
        assert "ri" not in set(features.summary.feature)
        assert any("No R/I" in m for m in warnings_of(features))

    features = peep_trial_features(trial, RI_12_6)
    assert features.config == RI_12_6  # kept with the results
    assert (value(features, "ri_peep_high"), value(features, "ri_peep_low")) == (12, 6)
    # The drop of PEEP of the formula is the measured one.
    measured_drop = trial.steps.peep_measured[0] - trial.steps.peep_measured[3]
    assert f"drop {measured_drop:.1f}" in " ".join(features.log)
    # The synthetic EELI does not follow the PEEP: no volume is lost, R/I = -1.
    assert value(features, "ri") == pytest.approx(-1, abs=0.05)
    volumes = features.summary[features.summary.feature == "recruited_volume_ml"]
    halves = volumes[volumes.region.isin(["ventral", "dorsal"])].value.sum()
    assert halves == pytest.approx(value(features, "recruited_volume_ml"), rel=0.02)


# --- the log: checks that passed (OK), missing inputs warned once ---


def warnings_of(features):
    return [m for m in features.log if m.startswith("WARNING: ")]


def test_a_complete_tidal_recording_has_no_warning_in_its_indices(tidal):
    assert not warnings_of(tidal)


def test_without_a_tidal_volume_one_warning():
    features = tidal_features(tidal_recording())
    assert [m for m in warnings_of(features) if "No tidal volume" in m] == [
        "WARNING: No tidal volume (none typed, no volume or flow curve): no values in mL."
    ]


def test_no_driving_pressure_is_warned_where_its_cause_is():
    context = RecordingContext(tidal_volume_ml=450)
    # No pressure at all: the warning is in the log of the preprocessing only.
    without_pressure = tidal_features(preprocess(synthetic_raw()[0], context=context))
    assert any(
        m.startswith("WARNING: No airway pressure")
        for m in without_pressure.recording.log
    )
    assert not warnings_of(without_pressure)
    # A pressure without an end-inspiratory pause: the warning is here.
    raw, _ = synthetic_raw()
    paw, _ = vcv_pressure([10] * 23, fs=FS, period=4.0, pause=0)
    raw.aux_signals = {"airway_pressure": paw[: len(raw.frames)].astype(np.float32)}
    without_plateau = tidal_features(preprocess(raw, context=context))
    assert any("No driving pressure" in m for m in warnings_of(without_plateau))


def test_a_trial_with_its_protocol_levels_has_its_checks():
    recording = synthetic_trial(context=RecordingContext(tidal_volume_ml=450))
    trial = segment_peep_trial(
        recording, SegmentationConfig(peep_levels=(12, 10, 8, 6))
    )
    features = peep_trial_features(trial)
    checks = [m for m in features.log if m.startswith("OK: ")]
    assert (
        "OK: Crossing PEEP and best PEEP on the PEEP levels of the protocol." in checks
    )
    assert "OK: Every step has a compliance." in checks
    # No R/I levels in the config: no R/I, said once, not a warning.
    assert not warnings_of(features)
    assert sum("No R/I" in m for m in features.log) == 1
