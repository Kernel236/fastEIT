"""Numpy structured dtype for Dräger `.eit` binary frame layout.

Each frame is exactly 5495 bytes. Layout reverse-engineered from PulmoVista
500 output and cross-referenced with EIDORS read_draeger_file (GPL,
understanding only - no code copied).

Key fields for EIT reconstruction:
    trans_A : float64[208] at offset 16    - primary transimpedance set
    trans_B : float64[208] at offset 2592  - reference transimpedance set

Transimpedance (Task 2.4.2):
    vv = calibration_factor[0] * trans_A - calibration_factor[1] * trans_B
"""

from __future__ import annotations

import numpy as np

# Preamble: 3 × int32 LE (format_version, sep_offset, unknown)
PREAMBLE_DTYPE = np.dtype("<i4")
PREAMBLE_N_FIELDS = 3

# Binary frame: 5495 bytes per frame
FRAME_EIT_DTYPE = np.dtype(
    [
        ("timestamp", "<f8"),  #    8 B - fraction of day
        ("unknown_f8", "<f8"),  #    8 B
        ("trans_A", "<f8", 208),  # 1664 B - primary transimpedances
        ("unknown_16a", "<f8", 16),  #  128 B
        ("injection_current", "<f8", 16),  #  128 B - ÷ 194326.3536 → Ampere
        ("unknown_16b", "<f8", 16),  #  128 B
        ("voltage_A", "<f8", 16),  #  128 B
        ("unknown_50", "<f8", 50),  #  400 B
        ("trans_B", "<f8", 208),  # 1664 B - reference transimpedances
        ("unknown_48", "<f8", 48),  #  384 B
        ("voltage_B", "<f8", 16),  #  128 B
        ("unknown_6", "<f8", 6),  #   48 B
        ("gugus", "<f8", 44),  #  352 B - ignored by EIDORS
        ("unknown_byte", "u1"),  #    1 B
        ("medibus", "<f4", 67),  #  268 B - 67 ventilator channels (.eit; .bin uses 52)
        ("event_text", "S30"),  #   30 B - ASCII, space-padded
        ("mixed", "u1", 24),  #   24 B
        ("frame_counter", "<u2"),  #    2 B
        ("padding", "u1", 2),  #    2 B
    ]
)
# Sanity check: dtype size must match known frame size
assert FRAME_EIT_DTYPE.itemsize == 5495, (
    f"FRAME_EIT_DTYPE size mismatch: {FRAME_EIT_DTYPE.itemsize} != 5495"
)

# Ventilator (Medibus) channels of the .eit frame with a known name: the .bin signal with the
# same name (MEDIBUS_EXT_FIELDS) has the same values, frame by frame. Found by matching the 67
# .eit channels with the .bin signals on 4 recordings with the ventilator connected; every
# signal matched one channel only. The order differs from the .bin. The other channels are not
# named (no data in those recordings). The PressurePod signals are not in the .eit frame.
EIT_MEDIBUS_CHANNELS: dict[str, int] = {
    "flow": 1,
    "airway_pressure": 3,
    "dynamic_compliance": 4,
    "resistance": 7,
    "minimal_pressure": 10,
    "mean_pressure": 12,
    "plateau_pressure": 13,
    "peep": 14,
    "spontaneous_minute_volume": 16,
    "peak_inspiratory_pressure": 19,
    "tidal_volume": 29,
    "negative_inspiratory_force": 31,
    "minute_volume": 38,
    "respiratory_rate": 41,
    "fio2": 50,
}
