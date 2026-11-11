"""Tests for the review figures (plot_segmentation, plot_breaths)."""

import matplotlib

matplotlib.use("Agg")  # no display in the tests

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fasteit.config import SegmentationConfig  # noqa: E402
from fasteit.features import tidal_features  # noqa: E402
from fasteit.models import Segment  # noqa: E402
from fasteit.plotting import plot_breaths, plot_segmentation  # noqa: E402
from fasteit.preprocessing import preprocess  # noqa: E402
from fasteit.segmentation import segment_peep_trial  # noqa: E402

from .preprocessing.test_breath_pressures import vcv_pressure  # noqa: E402
from .preprocessing.test_pipeline import FS, synthetic_raw  # noqa: E402
from .segmentation.test_pipeline import BASELINE_THEN_TRIAL, TRIAL_PHASE  # noqa: E402

BASELINE_PHASE = Segment(0, 40, category="phase", label="tidal")


@pytest.fixture(scope="module")
def raw():
    """200 s at 15 breaths/min: a baseline at PEEP 12, then a trial 16-14-12-10."""
    raw, _ = synthetic_raw(seconds=200)
    paw, _ = vcv_pressure(BASELINE_THEN_TRIAL, fs=FS, period=4.0)
    raw.aux_signals = {"airway_pressure_pod": paw.astype(np.float32)}
    return raw


@pytest.fixture(scope="module")
def recording(raw):
    return preprocess(raw, exclusions=[Segment(100, 110, label="movement")])


def legend_texts(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def test_segmentation_shows_every_segment_with_its_kind(raw):
    segments = [
        BASELINE_PHASE,
        TRIAL_PHASE,
        Segment(100, 110, label="movement"),
        Segment(60, 80, category="peep_step", parameters={"peep": 14}),
    ]
    fig = plot_segmentation(raw, segments, title="A, supine")
    top, bottom = fig.axes
    assert len(top.patches) == len(segments)  # one band each
    names = [text.get_text() for text in top.texts]
    assert {"tidal", "peep trial", "movement", "PEEP 14"} <= set(names)
    assert legend_texts(top) == [
        "phase: tidal", "phase: peep trial", "excluded", "PEEP step",
    ]  # fmt: skip
    assert bottom.get_ylabel() == "Paw (cmH2O)"
    plt.close(fig)


def test_segmentation_without_pressure_says_so(raw):
    no_pressure, _ = synthetic_raw(seconds=60)
    fig = plot_segmentation(no_pressure)
    assert fig.axes[1].texts[0].get_text() == "no airway pressure"
    plt.close(fig)


def test_segmentation_has_a_gap_where_a_frame_has_missing_pixels():
    draeger, _ = synthetic_raw(vendor="draeger", seconds=20)
    draeger.frames["pixels"][100, 10, 10] = -1000.0  # Draeger "no data"
    fig = plot_segmentation(draeger)
    signal = fig.axes[0].lines[0].get_ydata()
    assert np.isnan(signal[100])  # not a false dip
    assert np.isfinite(np.delete(signal, 100)).all()
    plt.close(fig)


def test_unknown_segment_categories_are_rejected(raw):
    with pytest.raises(ValueError, match="Unknown segment category"):
        plot_segmentation(raw, [Segment(0, 10, category="other")])


def test_breaths_of_a_peep_trial_show_the_analysed_ones_and_the_steps(recording):
    config = SegmentationConfig(peep_levels=(16, 14, 12, 10))
    trial = segment_peep_trial(recording, config, phase=TRIAL_PHASE)
    fig = plot_breaths(trial)
    top, bottom, image = fig.axes
    counts = dict(text.rsplit(" (", 1) for text in legend_texts(image))
    assert int(counts["used"].rstrip(")")) == trial.breaths.analysed.sum()
    assert int(counts["excluded"].rstrip(")")) == trial.breaths.excluded.sum() > 0
    assert [text.get_text() for text in top.texts] == ["16", "14", "12", "10"]
    assert bottom.get_ylabel() == "PEEP (cmH2O)"
    assert image.get_title() == "tidal image, lung mask"
    plt.close(fig)


def test_breaths_of_a_tidal_phase_are_used_only_inside_it(recording):
    features = tidal_features(recording, phase=BASELINE_PHASE)
    fig = plot_breaths(features)
    counts = dict(text.rsplit(" (", 1) for text in legend_texts(fig.axes[2]))
    in_phase = features.breaths.in_phase.to_numpy()
    expected = (recording.breaths.is_valid.to_numpy() & in_phase).sum()
    assert int(counts["used"].rstrip(")")) == expected
    assert int(counts["valid, not used"].rstrip(")")) > 0  # the trial part
    plt.close(fig)


def test_only_results_of_the_pipeline_are_accepted(raw):
    with pytest.raises(TypeError, match="preprocess"):
        plot_breaths(raw)
