"""Layout of the Draeger .bin frames and of the Medibus fields."""

import numpy as np
import pytest

from fasteit.parsers.draeger.bin.draeger_dtypes import (
    FRAME_BASE_DTYPE,
    FRAME_EXT_DTYPE,
    MEDIBUS_BASE_FIELDS,
    MEDIBUS_BASE_INDEX,
    MEDIBUS_EXT_FIELDS,
    MEDIBUS_EXT_INDEX,
)

# Fields shared by both frame formats: (name, numpy type, shape).
FRAME_FIELDS = [
    ("ts", "<f8", ()),  # time as a fraction of the day
    ("dummy", "<f4", ()),
    ("pixels", "<f4", (32, 32)),
    ("min_max_flag", "<i4", ()),
    ("event_marker", "<i4", ()),
    ("event_text", "S30", ()),
    ("timing_error", "<i4", ()),
]


@pytest.mark.parametrize(
    "dtype, frame_bytes, n_medibus",
    [(FRAME_BASE_DTYPE, 4358, 52), (FRAME_EXT_DTYPE, 4382, 58)],
)
def test_frame_layout(dtype, frame_bytes, n_medibus):
    assert dtype.itemsize == frame_bytes
    for name, numpy_type, shape in FRAME_FIELDS:
        assert dtype[name].base == np.dtype(numpy_type), name
        assert dtype[name].shape == shape, name
    assert dtype["medibus_data"].base == np.dtype("<f4")
    assert dtype["medibus_data"].shape == (n_medibus,)
    # A zero-filled buffer of one frame parses as one frame.
    frame = np.frombuffer(bytes(frame_bytes), dtype=dtype)
    assert frame.shape == (1,) and frame["medibus_data"].shape == (1, n_medibus)


def test_medibus_fields():
    assert (len(MEDIBUS_BASE_FIELDS), len(MEDIBUS_EXT_FIELDS)) == (52, 58)
    for name, unit, is_continuous in MEDIBUS_BASE_FIELDS:
        assert isinstance(name, str) and name
        assert isinstance(unit, str) and isinstance(is_continuous, bool)
    # The first 6 fields are waveforms, the next 46 values computed once per breath.
    continuous = [is_continuous for _, _, is_continuous in MEDIBUS_BASE_FIELDS]
    assert continuous == [True] * 6 + [False] * 46


@pytest.mark.parametrize(
    "index, name, position",
    [
        (MEDIBUS_BASE_INDEX, "airway_pressure", 0),
        (MEDIBUS_BASE_INDEX, "flow", 1),
        (MEDIBUS_BASE_INDEX, "volume", 2),
        (MEDIBUS_BASE_INDEX, "peep", 14),
        (MEDIBUS_BASE_INDEX, "time_at_low_pressure", 51),  # last field of the base
        (MEDIBUS_EXT_INDEX, "airway_pressure", 0),
        (MEDIBUS_EXT_INDEX, "peep", 14),
        (MEDIBUS_EXT_INDEX, "high_pressure", 51),  # first field of the extended format
        (MEDIBUS_EXT_INDEX, "low_pressure", 52),
        (MEDIBUS_EXT_INDEX, "time_at_low_pressure", 53),  # shifted from 51
    ],
)
def test_medibus_index(index, name, position):
    assert index[name] == position


def test_the_extended_format_has_the_pressurepod_channels():
    for name in ["airway_pressure_pod", "esophageal_pressure_pod",
                 "transpulmonary_pressure_pod", "gastric_pressure_pod"]:  # fmt: skip
        assert name in MEDIBUS_EXT_INDEX
