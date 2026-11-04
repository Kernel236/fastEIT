"""Tests for reconstruction.reconstruct - from an .eit to ReconstructedFrameData."""

from pathlib import Path

import numpy as np
import pytest

from fasteit.models import RawImpedanceData, ReconstructedFrameData
from fasteit.reconstruction.model import fit_model
from fasteit.reconstruction.reconstruct import reconstruct

from .test_model import recording

FS = 50.0
TEST_FILES = Path(__file__).parents[2] / "src" / "fasteit" / "test_files"


@pytest.fixture(scope="module")
def model():
    return fit_model([recording(seed) for seed in range(10, 14)])


def raw_eit(x):
    ts = 0.5 + np.arange(len(x)) / FS / 86400  # fraction of the day, as in Dräger files
    return RawImpedanceData(
        measurements=x,
        aux_signals={"timestamp": ts},
        fs=FS,
        filename="synthetic.eit",
        file_format="eit",
        vendor="draeger",
    )


class TestReconstruct:
    def test_shapes_and_timestamps(self, model):
        x, _ = recording(50)
        out = reconstruct(raw_eit(x), model)
        assert isinstance(out, ReconstructedFrameData)
        assert out.pixels.shape == (len(x), 32, 32)
        assert out.n_frames == len(x)
        np.testing.assert_array_equal(
            out.timestamps, raw_eit(x).aux_signals["timestamp"]
        )
        assert out.vendor == "draeger"
        assert out.metadata["lowpass_bpm"] == 50

    def test_like_gives_ts_and_ventilator_signals(self, model):
        x, _ = recording(51)
        frames = np.zeros(
            len(x),
            dtype=[
                ("ts", "<f8"),
                ("pixels", "<f4", (32, 32)),
                ("min_max_flag", "<i4"),
                ("event_marker", "<i4"),
            ],
        )
        frames["ts"] = 0.7 + np.arange(len(x)) / FS / 86400
        frames["event_marker"][100:] = 1
        paw = np.linspace(5, 15, len(x))
        like = ReconstructedFrameData(
            frames=frames, aux_signals={"airway_pressure": paw}, fs=FS, vendor="draeger"
        )
        out = reconstruct(raw_eit(x), model, like=like)
        np.testing.assert_array_equal(out.timestamps, frames["ts"])
        np.testing.assert_array_equal(out.event_markers, frames["event_marker"])
        np.testing.assert_array_equal(out.aux_signals["airway_pressure"], paw)
        assert "min_max_flag" not in out.frames.dtype.names  # Dräger's own breath marks

    def test_without_like_the_ventilator_signals_come_from_the_eit(self, model):
        x, _ = recording(54)
        raw = raw_eit(x)
        raw.aux_signals |= {"airway_pressure": np.linspace(5, 15, len(x)), "trans_A": x}
        out = reconstruct(raw, model)
        np.testing.assert_array_equal(
            out.aux_signals["airway_pressure"], raw.aux_signals["airway_pressure"]
        )
        assert "trans_A" not in out.aux_signals  # only the ventilator signals

    def test_lowpass_none_keeps_the_heart_beat(self, model):
        x, _ = recording(52, n_frames=1500)
        t = np.arange(len(x)) / FS
        x = (
            x * (1 + 0.01 * np.sin(2 * np.pi * 2.0 * t))[:, None]
        )  # 120/min on every channel

        def heart_power(images):
            spectrum = np.abs(
                np.fft.rfft(images.sum(axis=(1, 2)) - images.sum(axis=(1, 2)).mean())
            )
            freq = np.fft.rfftfreq(len(t), 1 / FS)
            return spectrum[np.argmin(np.abs(freq - 2.0))]

        raw = raw_eit(x)
        unfiltered = heart_power(reconstruct(raw, model, lowpass_bpm=None).pixels)
        filtered = heart_power(reconstruct(raw, model).pixels)
        assert filtered < 0.05 * unfiltered

    def test_too_short_for_the_filter_raises(self, model):
        x, _ = recording(53, n_frames=8)
        with pytest.raises(ValueError, match="too short"):
            reconstruct(raw_eit(x), model)


def _real_pair():
    """Shortest .eit/.bin pair of test_files with the same number of frames, or None."""
    from fasteit.parsers.loader import load_data

    pairs = []
    for e in sorted(TEST_FILES.glob("*.eit")):
        b = next(
            (b for b in TEST_FILES.glob("*.bin") if b.stem in (e.stem, e.stem + "_01")),
            None,
        )
        if b is not None:
            pairs.append((e.stat().st_size, e, b))
    for _, e, b in sorted(pairs):
        raw, like = load_data(e), load_data(b)
        if raw.n_frames == like.n_frames:
            return raw, like
    return None


@pytest.mark.slow
def test_real_pair_runs_through_preprocess():
    from fasteit.features import tidal_features
    from fasteit.preprocessing import preprocess

    pair = _real_pair()
    if pair is None:
        pytest.skip("no paired .eit/.bin in test_files")
    raw, like = pair
    y = like.pixels.reshape(like.n_frames, -1)
    model = fit_model([(raw.measurements, y)])
    recording_ml = preprocess(reconstruct(raw, model, like=like))
    features = tidal_features(recording_ml)
    assert len(features.breaths) > 0
