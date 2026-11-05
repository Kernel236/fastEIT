"""Tests for the tables of layer 4 (recording_tables, combine_tables, write_csv)."""

import numpy as np
import pandas as pd
import pytest

from fasteit.config import SegmentationConfig
from fasteit.export import KEYS, combine_tables, recording_tables, write_csv
from fasteit.features import peep_trial_features, tidal_features
from fasteit.models import RecordingContext, Segment
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

from ..preprocessing.test_breath_pressures import vcv_pressure
from ..preprocessing.test_pipeline import FS, synthetic_raw
from ..segmentation.test_pipeline import BASELINE_THEN_TRIAL, TRIAL_PHASE

CONTEXT = RecordingContext(patient_id="P01", label="T0", tidal_volume_ml=450)
LEVELS = SegmentationConfig(peep_levels=(16, 14, 12, 10))
BASELINE_PHASE = Segment(0, 40, category="phase", label="tidal")


def trial_recording(peeps):
    """200 s at 15 breaths/min with the PEEP of each breath in the airway pressure."""
    raw, _ = synthetic_raw(seconds=200)
    paw, _ = vcv_pressure(peeps, fs=FS, period=4.0)
    raw.aux_signals = {"airway_pressure_pod": paw.astype(np.float32)}
    return preprocess(raw, context=CONTEXT)


@pytest.fixture(scope="module")
def composite():
    """A baseline at PEEP 12, then a PEEP trial 16-14-12-10, in one recording."""
    recording = trial_recording(BASELINE_THEN_TRIAL)
    baseline = tidal_features(recording, phase=BASELINE_PHASE)
    trial = segment_peep_trial(recording, LEVELS, phase=TRIAL_PHASE)
    return baseline, peep_trial_features(trial)


@pytest.fixture(scope="module")
def tables(composite):
    baseline, trial = composite
    return recording_tables(baseline, phase="baseline"), recording_tables(trial)


# --- one analysis ---


def test_every_row_starts_with_the_keys(tables):
    for analysis in tables:
        for table in analysis.values():
            assert list(table.columns[:3]) == KEYS
    baseline, trial = tables
    assert baseline["recordings"][KEYS].iloc[0].tolist() == ["P01", "T0", "baseline"]
    assert trial["recordings"][KEYS].iloc[0].tolist() == ["P01", "T0", "peep_trial"]


def test_label_and_phase_can_be_changed_at_export(composite):
    tables = recording_tables(composite[0], label="T1", phase="recovery")
    assert tables["breaths"].label.eq("T1").all()
    assert tables["breaths"].phase.eq("recovery").all()


def test_each_phase_has_only_its_breaths(composite, tables):
    baseline, trial = tables
    assert (baseline["breaths"].end_time <= BASELINE_PHASE.end).all()
    assert (trial["breaths"].start_time >= TRIAL_PHASE.start).all()
    # One recording, two analyses: no breath is in both.
    both = set(baseline["breaths"].breath_id) & set(trial["breaths"].breath_id)
    assert not both


def test_used_breaths_are_those_of_the_indices(composite, tables):
    baseline, trial = tables
    assert baseline["breaths"].used.equals(baseline["breaths"].is_valid)
    analysed = composite[1].trial.analysed_breaths.breath_id
    assert set(trial["breaths"].breath_id[trial["breaths"].used]) == set(analysed)
    assert trial["recordings"].n_used.iloc[0] == len(analysed)


def test_a_breath_row_joins_layers_2_and_3(composite, tables):
    trial = tables[1]["breaths"]
    for column in ["start_time", "is_valid", "peep", "driving_pressure",  # layer 2
                   "peep_step", "peep_set",                              # layer 2b
                   "eeli", "tiv", "compliance_ml"]:  # layer 3  # fmt: skip
        assert column in trial.columns
    assert "start_idx" not in trial.columns and "analysed" not in trial.columns
    in_steps = trial[trial.peep_step >= 0]
    assert set(in_steps.peep_set) == {16, 14, 12, 10}
    assert trial.peep_set[trial.peep_step < 0].isna().all()


def test_regions_are_columns_with_the_values_of_the_long_table(composite, tables):
    features, breaths = composite[1], tables[1]["breaths"]
    long = features.regions.set_index(["breath_id", "region"])
    for breath_id in breaths.breath_id[:5]:
        row = breaths[breaths.breath_id == breath_id].iloc[0]
        for region in ["ventral", "left"]:
            expected = long.loc[(breath_id, region), "tiv"]
            assert row[f"tiv__{region}"] == pytest.approx(expected, nan_ok=True)


def test_only_the_halves_of_the_lung_are_exported(tables):
    for table in [tables[1]["breaths"], tables[1]["steps"], tables[1]["recordings"]]:
        regions = {c.split("__")[1] for c in table.columns if "__" in c}
        assert regions <= {"global", "ventral", "dorsal", "right", "left"}
    regional = [c for c in tables[1]["breaths"].columns if "__" in c]
    assert len(regional) == 6 * 4  # 6 indices x 4 halves


def test_the_step_table_has_the_curve_of_the_trial(composite, tables):
    steps = tables[1]["steps"]
    assert steps.peep_set.tolist() == [16, 14, 12, 10]
    for column in ["driving_pressure", "tidal_volume", "compliance", "collapse",
                   "overdistension", "compliance__ventral", "share__dorsal"]:  # fmt: skip
        assert column in steps.columns
    np.testing.assert_allclose(steps.collapse, composite[1].steps.collapse)


def test_the_recording_row_has_only_what_needs_the_methods(composite, tables):
    baseline, trial = tables[0]["recordings"], tables[1]["recordings"]
    # Tidal: file and breaths only; medians are made from breaths.csv at analysis time.
    assert not [c for c in baseline.columns if "__" in c]
    # PEEP trial: crossing PEEP, best PEEP, R/I (global and by half), no step by level.
    summary = composite[1].summary.set_index(["feature", "region"])
    assert trial["crossing_peep__global"].iloc[0] == pytest.approx(
        summary.loc[("crossing_peep", "global"), "value"]
    )
    assert not [c for c in trial.columns if "__peep" in c]


def test_settings_and_log_of_every_layer(composite, tables):
    settings, log = tables[1]["settings"], tables[1]["log"]
    assert set(settings.section) == {
        "fasteit", "preprocessing", "context", "segmentation", "analysis",
    }  # fmt: skip
    levels = settings[settings.parameter == "peep_levels"].value.iloc[0]
    assert levels == "(16.0, 14.0, 12.0, 10.0)"
    assert set(log.layer) == {"preprocessing", "segmentation", "indices"}
    assert len(log) == sum(
        len(messages)
        for messages in [
            composite[1].trial.recording.log,
            composite[1].trial.log,
            composite[1].log,
        ]
    )


def test_log_rows_have_a_level_and_the_message_without_its_prefix(composite, tables):
    log = tables[1]["log"]
    assert list(log.columns[-3:]) == ["layer", "level", "message"]
    assert set(log.level) <= {"ok", "info", "warning"}
    written = [
        *composite[1].trial.recording.log,
        *composite[1].trial.log,
        *composite[1].log,
    ]
    warnings = [
        m.removeprefix("WARNING: ") for m in written if m.startswith("WARNING: ")
    ]
    assert log[log.level == "warning"].message.tolist() == warnings
    assert not log.message.str.startswith(("OK: ", "WARNING: ")).any()


def test_only_layer_3_results_are_accepted(composite):
    with pytest.raises(TypeError, match="tidal_features"):
        recording_tables(composite[1].trial)


# --- a study ---


def test_a_study_has_the_rows_of_every_analysis(tables):
    study = combine_tables(list(tables))
    assert len(study["recordings"]) == 2
    assert len(study["breaths"]) == sum(len(t["breaths"]) for t in tables)
    assert len(study["steps"]) == len(tables[1]["steps"])  # tidal has none
    # Columns of one protocol are empty for the other.
    assert study["recordings"].crossing_peep__global.isna().iloc[0]


def test_two_analyses_with_the_same_keys_are_rejected(tables):
    with pytest.raises(ValueError, match="label='T0', phase='baseline'"):
        combine_tables([tables[0], tables[0]])
    with pytest.raises(ValueError, match="No analysis"):
        combine_tables([])


def test_booleans_of_combined_tables_are_written_as_numbers(tmp_path):
    # A boolean column missing in one of the combined tables holds True, False, NaN.
    table = pd.DataFrame({"flag": [True, np.nan, False], "label": ["a", None, "b"]})
    path = write_csv({"t": table}, tmp_path)["t"]
    assert path.read_text().splitlines() == ["flag,label", "1,a", ",", "0,b"]


def test_csv_files_for_r_and_spss(tables, tmp_path):
    paths = write_csv(combine_tables(list(tables)), tmp_path / "results")
    assert sorted(paths) == ["breaths", "log", "recordings", "settings", "steps"]
    breaths = pd.read_csv(paths["breaths"])
    assert len(breaths) == sum(len(t["breaths"]) for t in tables)
    assert set(breaths.used.unique()) <= {0, 1}  # booleans as 1 / 0
    assert breaths.exclusion_label.isna().all()  # missing: empty cell
    steps = pd.read_csv(paths["steps"])
    assert set(steps.enough.dropna().unique()) <= {0, 1}
