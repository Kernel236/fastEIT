"""Numpy structured dtypes for Dräger PulmoVista 500 .bin frame layouts.

Each .bin file is a flat sequence of frames. Two frame sizes exist:
- 4358 bytes: base format (FRAME_BASE_DTYPE)
- 4382 bytes: extended format with PressurePod (FRAME_EXT_DTYPE, +24 bytes = 6 extra Medibus fields)

Layout confirmed from three independent sources:
1. EIDORS read_draeger_bin (MATLAB, GPL — format study only, no code copied)
2. eitprocessing draeger.py v0.3 (Apache-2.0)
3. Shared professional MATLAB analysis scripts
"""

from __future__ import annotations

import numpy as np

# ── Base frame (4358 bytes) ────────────────────────────────────────────────────

FRAME_BASE_DTYPE = np.dtype(
    [
        ("ts", "<f8"),  #   8 bytes — time as fraction of day (float64 LE)
        ("dummy", "<f4"),  #   4 bytes — bytes 8–11, unused
        (
            "pixels",
            "<f4",
            (32, 32),
        ),  # 4096 bytes — reconstructed 32×32 impedance image, C-order
        (
            "min_max_flag",
            "<i4",
        ),  #   4 bytes — breath phase: +1=insp peak, -1=exp trough, 0=none
        (
            "event_marker",
            "<i4",
        ),  #   4 bytes — event counter (increments on each new event)
        ("event_text", "S30"),  #  30 bytes — event text string (null-padded ASCII)
        ("timing_error", "<i4"),  #   4 bytes — timing error flag (0 = ok)
        ("medibus_data", "<f4", (52,)),  # 208 bytes — 52 Medibus waveform values
    ]
)
# Total: 8+4+4096+4+4+30+4+208 = 4358 bytes

# ── Extended frame (4382 bytes) — with PressurePod ────────────────────────────

FRAME_EXT_DTYPE = np.dtype(
    [
        ("ts", "<f8"),  #   8 bytes — time as fraction of day (float64 LE)
        ("dummy", "<f4"),  #   4 bytes — bytes 8–11, unused
        (
            "pixels",
            "<f4",
            (32, 32),
        ),  # 4096 bytes — reconstructed 32×32 impedance image, C-order
        ("min_max_flag", "<i4"),  #   4 bytes — breath phase marker
        ("event_marker", "<i4"),  #   4 bytes — event counter
        ("event_text", "S30"),  #  30 bytes — event text string
        ("timing_error", "<i4"),  #   4 bytes — timing error flag
        (
            "medibus_data",
            "<f4",
            (58,),
        ),  # 232 bytes — 58 Medibus values (52 base + 6 PressurePod)
    ]
)
# Total: 8+4+4096+4+4+30+4+232 = 4382 bytes (= 4358 + 24 = base + 6 × float32)

# ── Medibus field definitions ─────────────────────────────────────────────────
# Each entry: (snake_case_name, unit, is_continuous_waveform)
# Index in this list == index in medibus_data array.

MEDIBUS_FIELDS: list[tuple[str, str, bool]] = [
    # idx 0-5: continuous waveforms (sampled every frame at 50 Hz)
    ("airway_pressure", "mbar", True),
    ("flow", "L/min", True),
    ("volume", "mL", True),
    ("co2_pct", "%", True),
    ("co2_kpa", "kPa", True),
    ("co2_mmhg", "mmHg", True),
    # idx 6-51: breath-averaged ventilator parameters (updated once per breath)
    ("dynamic_compliance", "mL/mbar", False),
    ("resistance", "mbar/L/s", False),
    ("r_squared", "", False),
    ("spontaneous_inspiratory_time", "s", False),
    ("minimal_pressure", "mbar", False),
    ("p0_1", "mbar", False),
    ("mean_pressure", "mbar", False),
    ("plateau_pressure", "mbar", False),
    ("peep", "mbar", False),
    ("intrinsic_peep", "mbar", False),
    ("mandatory_respiratory_rate", "/min", False),
    ("mandatory_minute_volume", "L/min", False),
    ("peak_inspiratory_pressure", "mbar", False),
    ("mandatory_tidal_volume", "L", False),
    ("spontaneous_tidal_volume", "L", False),
    ("trapped_volume", "mL", False),
    ("mandatory_expiratory_tidal_volume", "mL", False),
    ("spontaneous_expiratory_tidal_volume", "mL", False),
    ("mandatory_inspiratory_tidal_volume", "mL", False),
    ("tidal_volume", "mL", False),
    ("spontaneous_inspiratory_tidal_volume", "mL", False),
    ("negative_inspiratory_force", "mbar", False),
    ("leak_minute_volume", "L/min", False),
    ("leak_percentage", "%", False),
    ("spontaneous_respiratory_rate", "/min", False),
    ("pct_spontaneous_minute_volume", "%", False),
    ("spontaneous_minute_volume", "L/min", False),
    ("minute_volume", "L/min", False),
    ("airway_temperature", "degC", False),
    ("rapid_shallow_breathing_index", "1/min/L", False),
    ("respiratory_rate", "/min", False),
    ("ie_ratio", "", False),
    ("co2_flow", "mL/min", False),
    ("dead_space_volume", "mL", False),
    ("pct_dead_space_expiratory_tv", "%", False),
    ("etco2_pct", "%", False),
    ("etco2_kpa", "kPa", False),
    ("etco2_mmhg", "mmHg", False),
    ("fio2", "%", False),
    ("spontaneous_ie_ratio", "", False),
    ("elastance", "mbar/L", False),
    ("time_constant", "s", False),
    ("upper_20pct_compliance_ratio", "", False),
    ("end_inspiratory_pressure", "mbar", False),
    ("expiratory_tidal_volume", "mL", False),
    ("time_at_low_pressure", "s", False),
]  # 52 fields

# 6 extra fields present only in the PressurePod (extended) format
_MEDIBUS_EXT_EXTRA: list[tuple[str, str, bool]] = [
    ("high_pressure", "mbar", False),
    ("low_pressure", "mbar", False),
    ("time_at_low_pressure_pod", "s", False),
    ("airway_pressure_pod", "mbar", True),
    ("esophageal_pressure_pod", "mbar", True),
    ("transpulmonary_pressure_pod", "mbar", True),
]

MEDIBUS_EXT_FIELDS: list[tuple[str, str, bool]] = MEDIBUS_FIELDS + _MEDIBUS_EXT_EXTRA
# 58 fields

# ── Lookup dicts: name → index ─────────────────────────────────────────────────
# These dicts let you find a field's position in the medibus_data array by name.
# Example: MEDIBUS_INDEX["peep"] == 14
#          frames["medibus_data"][:, MEDIBUS_INDEX["peep"]]  → PEEP waveform

MEDIBUS_INDEX: dict[str, int] = {}
for _i, (_name, _unit, _continuous) in enumerate(MEDIBUS_FIELDS):
    MEDIBUS_INDEX[_name] = _i

MEDIBUS_EXT_INDEX: dict[str, int] = {}
for _i, (_name, _unit, _continuous) in enumerate(MEDIBUS_EXT_FIELDS):
    MEDIBUS_EXT_INDEX[_name] = _i
