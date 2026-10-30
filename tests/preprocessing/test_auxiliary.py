"""Tests for the auxiliary signals."""

import numpy as np

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.preprocessing.auxiliary import auxiliary_signals

NOT_CONNECTED = np.array([0xFF7FC99E], dtype=np.uint32).view(np.float32)[0]


def draeger_raw():
    frames = np.zeros(4, dtype=[("ts", "f8"), ("min_max_flag", "i4")])
    frames["min_max_flag"] = [0, 1, 0, -1]
    aux = {
        "airway_pressure_pod": np.array([5.0, 12.0, -1000.0, 5.0], dtype=np.float32),
        "peep": np.full(4, -1000.0, dtype=np.float32),  # no ventilator connected
        "flow": np.full(4, NOT_CONNECTED, dtype=np.float32),
    }
    return ReconstructedFrameData(
        fs=50.0, vendor="draeger", frames=frames, aux_signals=aux
    )


def test_draeger_no_data_values_become_nan_and_empty_channels_are_dropped():
    aux = auxiliary_signals(draeger_raw())
    assert list(aux.columns) == ["airway_pressure_pod", "min_max_flag"]
    np.testing.assert_array_equal(aux.airway_pressure_pod, [5.0, 12.0, np.nan, 5.0])


def test_draeger_breath_markers_are_kept_as_integers():
    aux = auxiliary_signals(draeger_raw())
    assert aux.min_max_flag.tolist() == [0, 1, 0, -1]
    assert aux.min_max_flag.dtype.kind == "i"


def test_the_parsed_signals_are_not_modified():
    raw = draeger_raw()
    auxiliary_signals(raw)
    assert raw.aux_signals["airway_pressure_pod"][2] == -1000.0


def test_timpel_channels_and_flags_pass_through():
    frames = np.zeros(3, dtype=[("ts", "f8")])
    aux = {
        "airway_pressure": np.array([4.0, np.nan, 6.0], dtype=np.float32),
        "qrs_flag": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    raw = ReconstructedFrameData(
        fs=50.0, vendor="timpel", frames=frames, aux_signals=aux
    )
    result = auxiliary_signals(raw)
    assert list(result.columns) == ["airway_pressure", "qrs_flag"]
    assert result.qrs_flag.tolist() == [0.0, 1.0, 0.0]


def test_no_auxiliary_signals_give_an_empty_table_with_one_row_per_frame():
    frames = np.zeros(5, dtype=[("ts", "f8")])
    raw = ReconstructedFrameData(fs=50.0, vendor="timpel", frames=frames)
    result = auxiliary_signals(raw)
    assert result.shape == (5, 0)
