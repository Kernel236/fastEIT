"""Layout of the Draeger .eit frames."""

import numpy as np
import pytest

from fasteit.parsers.draeger.eit.eit_dtypes import (
    FRAME_EIT_DTYPE,
    PREAMBLE_DTYPE,
    PREAMBLE_N_FIELDS,
)

FRAME_BYTES = 5495
# (name, numpy type, shape) of the fields read by the parser.
FRAME_FIELDS = [
    ("timestamp", "<f8", ()),
    ("trans_A", "<f8", (208,)),  # 16 injections x 13 measurements
    ("trans_B", "<f8", (208,)),
    ("injection_current", "<f8", (16,)),
    ("voltage_A", "<f8", (16,)),
    ("voltage_B", "<f8", (16,)),
    ("medibus", "<f4", (67,)),
    ("frame_counter", "<u2", ()),
    ("event_text", "S30", ()),
]


def test_preamble():
    assert PREAMBLE_DTYPE == np.dtype("<i4") and PREAMBLE_N_FIELDS == 3


@pytest.mark.parametrize("name, numpy_type, shape", FRAME_FIELDS)
def test_frame_field(name, numpy_type, shape):
    assert FRAME_EIT_DTYPE[name].base == np.dtype(numpy_type)
    assert FRAME_EIT_DTYPE[name].shape == shape


def test_frame_size_and_offsets():
    assert FRAME_EIT_DTYPE.itemsize == FRAME_BYTES
    assert FRAME_EIT_DTYPE.fields["trans_A"][1] == 16  # after timestamp and 8 bytes
    assert FRAME_EIT_DTYPE.fields["trans_B"][1] == 2592  # from the reverse engineering


@pytest.mark.parametrize("n_frames", [1, 5])
def test_zero_filled_frames_parse(n_frames):
    frames = np.frombuffer(bytes(FRAME_BYTES * n_frames), dtype=FRAME_EIT_DTYPE)
    assert frames.shape == (n_frames,)
    assert frames["trans_A"].shape == (n_frames, 208)


def test_named_medibus_channels_are_unique_and_known():
    from fasteit.parsers.draeger.bin.draeger_dtypes import MEDIBUS_EXT_FIELDS
    from fasteit.parsers.draeger.eit.eit_dtypes import EIT_MEDIBUS_CHANNELS

    assert len(EIT_MEDIBUS_CHANNELS) == 15
    assert len(set(EIT_MEDIBUS_CHANNELS.values())) == 15
    assert all(0 <= i < 67 for i in EIT_MEDIBUS_CHANNELS.values())
    assert set(EIT_MEDIBUS_CHANNELS) <= {field[0] for field in MEDIBUS_EXT_FIELDS}
