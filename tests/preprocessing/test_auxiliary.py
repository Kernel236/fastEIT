"""Tests for the auxiliary signals."""

import numpy as np
import pytest

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.draeger.bin.draeger_dtypes import MEDIBUS_UNITS
from fasteit.parsers.timpel.timpel_dtypes import TIMPEL_AUX_FIELDS, TIMPEL_AUX_UNITS
from fasteit.preprocessing.auxiliary import (
    CMH2O_PER_MBAR,
    COMMON_UNITS,
    auxiliary_signals,
    common_unit,
)

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
    aux, units = auxiliary_signals(draeger_raw())
    assert list(aux.columns) == ["airway_pressure_pod", "min_max_flag"]
    assert units == {"airway_pressure_pod": "cmH2O", "min_max_flag": ""}
    assert np.isnan(aux.airway_pressure_pod[2])


def test_draeger_pressures_are_converted_from_mbar_to_cmh2o():
    aux, _ = auxiliary_signals(draeger_raw())
    np.testing.assert_allclose(
        aux.airway_pressure_pod[[0, 1]], [5.099, 12.237], atol=1e-3
    )
    assert CMH2O_PER_MBAR == pytest.approx(1.0197, abs=1e-4)


def test_draeger_breath_markers_are_kept_as_integers():
    aux, _ = auxiliary_signals(draeger_raw())
    assert aux.min_max_flag.tolist() == [0, 1, 0, -1]
    assert aux.min_max_flag.dtype.kind == "i"


def test_the_parsed_signals_are_not_modified():
    raw = draeger_raw()
    auxiliary_signals(raw)
    assert raw.aux_signals["airway_pressure_pod"][2] == -1000.0
    assert raw.aux_signals["airway_pressure_pod"][1] == 12.0


def test_draeger_ventilator_channels_get_common_units():
    frames = np.zeros(2, dtype=[("ts", "f8")])
    aux = {
        "flow": np.array([30.0, -60.0], dtype=np.float32),  # L/min
        "volume": np.array([0.0, 450.0], dtype=np.float32),  # mL
        "mandatory_tidal_volume": np.array([0.45, 0.45], dtype=np.float32),  # L
        "minute_volume": np.array([9.0, 9.0], dtype=np.float32),  # L/min
        "dynamic_compliance": np.array([40.0, 40.0], dtype=np.float32),  # mL/mbar
    }
    raw = ReconstructedFrameData(
        fs=50.0, vendor="draeger", frames=frames, aux_signals=aux
    )
    result, units = auxiliary_signals(raw)
    np.testing.assert_allclose(result.flow, [0.5, -1.0])
    np.testing.assert_allclose(result.volume, [0.0, 450.0])
    np.testing.assert_allclose(result.mandatory_tidal_volume, 450.0, rtol=1e-6)
    np.testing.assert_allclose(result.minute_volume, 9.0)
    np.testing.assert_allclose(result.dynamic_compliance, 40.0 / CMH2O_PER_MBAR)
    assert units == {
        "flow": "L/s",
        "volume": "mL",
        "mandatory_tidal_volume": "mL",
        "minute_volume": "L/min",
        "dynamic_compliance": "mL/cmH2O",
    }


def test_timpel_channels_get_common_units_and_flags_pass_through():
    frames = np.zeros(3, dtype=[("ts", "f8")])
    aux = {
        "airway_pressure": np.array([4.0, np.nan, 6.0], dtype=np.float32),
        "flow": np.array([0.5, 0.0, -0.5], dtype=np.float32),
        "volume": np.array([0.0, 0.45, 0.0], dtype=np.float32),
        "qrs_flag": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    raw = ReconstructedFrameData(
        fs=50.0, vendor="timpel", frames=frames, aux_signals=aux
    )
    result, units = auxiliary_signals(raw)
    assert list(result.columns) == ["airway_pressure", "flow", "volume", "qrs_flag"]
    np.testing.assert_allclose(result.airway_pressure, [4.0, np.nan, 6.0])
    np.testing.assert_allclose(result.flow, [0.5, 0.0, -0.5])
    np.testing.assert_allclose(result.volume, [0.0, 450.0, 0.0], rtol=1e-6)
    assert result.qrs_flag.tolist() == [0.0, 1.0, 0.0]
    assert units == {
        "airway_pressure": "cmH2O",
        "flow": "L/s",
        "volume": "mL",
        "qrs_flag": "",
    }


def test_every_parser_unit_with_mbar_or_litres_is_converted():
    for name, unit in MEDIBUS_UNITS.items():
        if "mbar" in unit or unit == "L":
            assert unit in COMMON_UNITS, name
    assert set(TIMPEL_AUX_UNITS) == set(TIMPEL_AUX_FIELDS)


def test_resistance_and_elastance_are_converted():
    assert common_unit("resistance", "mbar/L/s") == ("cmH2O/L/s", CMH2O_PER_MBAR)
    assert common_unit("elastance", "mbar/L") == ("cmH2O/L", CMH2O_PER_MBAR)
    assert common_unit("minute_volume", "L/min") == ("L/min", 1.0)


def test_an_unknown_channel_is_kept_as_it_is():
    assert common_unit("something", None) == (None, 1.0)
    assert common_unit("co2_kpa", "kPa") == ("kPa", 1.0)


def test_no_auxiliary_signals_give_an_empty_table_with_one_row_per_frame():
    frames = np.zeros(5, dtype=[("ts", "f8")])
    raw = ReconstructedFrameData(fs=50.0, vendor="timpel", frames=frames)
    result, units = auxiliary_signals(raw)
    assert result.shape == (5, 0) and units == {}
