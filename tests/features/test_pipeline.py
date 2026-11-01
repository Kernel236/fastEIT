"""Tests for tidal_features() and peep_trial_features()."""

import numpy as np
import pytest

from fasteit.features import peep_trial_features, tidal_features
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

from ..preprocessing.test_breath_pressures import vcv_pressure
from ..preprocessing.test_breath_quality import TEST_FILES
from ..preprocessing.test_pipeline import FS, synthetic_raw
from ..preprocessing.test_tidal_volume import vcv_flow, volume_curve
from ..segmentation.test_pipeline import synthetic_trial


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
    b = tidal.breaths
    assert len(b) == len(tidal.recording.breaths)
    assert (b.tidal_volume_source == "typed_recording").all()
    np.testing.assert_allclose(middle(b.driving_pressure), 12, atol=0.2)
    np.testing.assert_allclose(middle(b.compliance_ml), 450 / 12, rtol=0.02)
    np.testing.assert_allclose(b.compliance, b.tiv / b.driving_pressure)
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
        right.compliance_ml, right.share * 450 / tidal.breaths.driving_pressure.values
    )


def test_the_summary_is_over_the_valid_breaths(tidal):
    s = tidal.summary
    row = s[(s.region == "global") & (s.feature == "compliance_ml")].iloc[0]
    assert row["median"] == pytest.approx(450 / 12, rel=0.02)
    assert row.q1 <= row["median"] <= row.q3
    tiv = s[(s.region == "global") & (s.feature == "tiv")].iloc[0]
    assert tiv.n_breaths == tidal.breaths.is_valid.sum()
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
    np.testing.assert_allclose(middle(b.tidal_volume), 450, atol=2)


def test_the_tidal_phase_limits_the_summary():
    recording = tidal_recording()
    features = tidal_features(
        recording, phase=Segment(40, 90, category="phase", label="tidal")
    )
    inside = features.breaths[features.breaths.in_phase]
    assert (inside.start_time >= 40).all()
    tiv = features.summary.query("region == 'global' and feature == 'tiv'").iloc[0]
    assert tiv.n_breaths == inside.is_valid.sum()
    with pytest.raises(ValueError, match="tidal"):
        tidal_features(
            recording, phase=Segment(40, 90, category="phase", label="peep_trial")
        )


# --- PEEP trials ---


@pytest.fixture(scope="module")
def trial_features():
    recording = synthetic_trial(context=RecordingContext(tidal_volume_ml=450))
    return peep_trial_features(segment_peep_trial(recording))


def test_one_row_per_step_with_compliance_and_changes(trial_features):
    s = trial_features.steps
    np.testing.assert_allclose(s.peep, [12, 10, 8, 6], atol=0.1)
    assert (s.n_analysed > 0).all()
    np.testing.assert_allclose(s.compliance_ml, 450 / 12, rtol=0.02)
    np.testing.assert_allclose(s.compliance, s.tiv / s.driving_pressure)
    assert s.compliance_pct_of_best.max() == pytest.approx(100)
    # The same ventilation at every step: no collapse or overdistension to speak of.
    assert s.collapse.max() < 2 and s.overdistension.max() < 2
    assert np.isnan(s.d_eeli_tiv_vs_previous[0]) and s.d_eeli_ml_vs_first[0] == 0
    np.testing.assert_allclose(
        s.d_eeli_ml_vs_first, s.d_eeli_tiv_vs_first * s.tidal_volume[0]
    )


def test_step_regions_and_summary(trial_features):
    r = trial_features.step_regions
    halves = r[r.region.isin(["ventral", "dorsal"])].groupby("step")
    np.testing.assert_allclose(halves.share.sum(), 1, atol=0.02)
    np.testing.assert_allclose(halves.tiv_ml.sum(), 450, rtol=0.02)
    summary = trial_features.summary.iloc[0]
    assert summary.n_steps == 4
    assert set(trial_features.breaths.columns) >= {"peep_step", "analysed"}


def test_a_tidal_volume_typed_for_a_step_goes_to_its_breaths():
    trial = segment_peep_trial(synthetic_trial(), tidal_volumes=[400, 420, 440, 460])
    features = peep_trial_features(trial)
    b = features.breaths
    in_step_1 = b[b.peep_step == 1]
    assert (in_step_1.tidal_volume == 420).all() and (
        in_step_1.tidal_volume_source == "typed"
    ).all()
    assert b[b.peep_step == -1].tidal_volume.isna().all()
    np.testing.assert_allclose(
        features.steps.compliance_ml,
        np.array([400, 420, 440, 460]) / features.steps.driving_pressure,
    )


# --- real PEEP trials (notebook 07, sections 3-4) ---

SET_VT = {"A": 550, "B": 400}
NOTEBOOK_07 = {  # crossing PEEP, best PEEP of Vt / driving, change of EELI in mL at the end
    "A_park_bench": (4.7, 6.2, -1734),
    "A_supine": (6.6, 6.2, -2419),
    "B_park_bench": (10.1, 8.5, -1441),
    "B_supine": (12.9, 12.7, -910),
}


@pytest.mark.slow  # about 25 s
@pytest.mark.skipif(
    not all((TEST_FILES / f"PEEP_trial_{n}.bin").exists() for n in NOTEBOOK_07),
    reason="real PEEP trials not available",
)
def test_real_peep_trials_give_the_indices_of_notebook_07():
    for name, (crossing, best_ml, eeli_ml) in NOTEBOOK_07.items():
        context = RecordingContext(
            protocol="peep_trial", tidal_volume_ml=SET_VT[name[0]]
        )
        recording = preprocess(
            load_data(TEST_FILES / f"PEEP_trial_{name}.bin"), context=context
        )
        features = peep_trial_features(segment_peep_trial(recording))
        summary = features.summary.iloc[0]
        assert summary.crossing_peep == pytest.approx(crossing, abs=0.05), name
        assert summary.best_peep_compliance_ml == pytest.approx(best_ml, abs=0.05), name
        last = features.steps.d_eeli_ml_vs_first.iloc[-1]
        assert last == pytest.approx(eeli_ml, abs=1), name
