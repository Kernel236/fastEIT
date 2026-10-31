"""Auxiliary signals recorded with the images: pressures, flow, volume, device markers.

The parsers keep them as they are in the file (``raw.aux_signals``): for Draeger
``.bin`` the channels that were not available hold "no data" values (-1000, or a
bit pattern read as about -3.4e38) instead of NaN, and each vendor has its own units.
Here they are cleaned once, so the next layers read every vendor in the same way:
"no data" as NaN and the same units, pressures in cmH2O, flow in L/s, volumes in mL.

The signals are not filtered and not filled: a missing value stays NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.draeger.bin.bin_utils import replace_draeger_sentinels
from fasteit.parsers.draeger.bin.draeger_dtypes import MEDIBUS_UNITS
from fasteit.parsers.timpel.timpel_dtypes import TIMPEL_AUX_UNITS

# Draeger per-frame device fields kept with the auxiliary signals.
DRAEGER_MARKERS = ("min_max_flag", "event_marker")

# 1 mbar = 100 Pa and 1 cmH2O = 98.0665 Pa, so 1 mbar = 1.0197 cmH2O.
CMH2O_PER_MBAR = 100 / 98.0665
ML_PER_L = 1000.0
# Unit given by the parser -> common unit and the factor to multiply by.
COMMON_UNITS = {
    "mbar": ("cmH2O", CMH2O_PER_MBAR),
    "mbar/L": ("cmH2O/L", CMH2O_PER_MBAR),
    "mbar/L/s": ("cmH2O/L/s", CMH2O_PER_MBAR),
    "mL/mbar": ("mL/cmH2O", 1 / CMH2O_PER_MBAR),
    "L": ("mL", ML_PER_L),
}
# The flow waveform in L/s (Draeger gives it in L/min); minute volumes stay in L/min.
FLOW = "flow"
FLOW_UNIT = ("L/s", 1 / 60)


def common_unit(name: str, unit: str | None) -> tuple[str | None, float]:
    """Common unit of a channel and the factor that converts it from its parser unit.

    Args:
        name: Channel name, e.g. ``"airway_pressure_pod"``.
        unit: Unit given by the parser (None if unknown).

    Returns:
        The common unit and the factor; unknown or other units are kept (factor 1).
    """
    if name == FLOW and unit == "L/min":
        return FLOW_UNIT
    return COMMON_UNITS.get(unit, (unit, 1.0))


def auxiliary_signals(
    raw: ReconstructedFrameData,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """One column per auxiliary channel that has data, one row per frame, and its unit.

    - Draeger "no data" values become NaN (Timpel channels are already NaN from the
      parser);
    - channels without any value are left out (e.g. Medibus when no ventilator was
      connected);
    - units become common to every vendor: pressures in cmH2O (Draeger gives mbar),
      flow in L/s, volumes in mL (Timpel gives L), and the same for the values derived
      from them (e.g. compliance in mL/cmH2O); other units are kept;
    - Draeger ``min_max_flag`` (+1 end of inspiration, -1 end of expiration, 0
      neither) and ``event_marker`` are added; Timpel ``min_flag``, ``max_flag`` and
      ``qrs_flag`` are already among its channels.

    Rows are frame numbers, the same as ``start_idx``, ``peak_idx`` and ``end_idx``
    of the breath table. The units of the parsers are in
    ``fasteit.parsers.draeger.bin.draeger_dtypes`` and ``timpel_dtypes``.

    Args:
        raw: Parsed recording; never modified.

    Returns:
        DataFrame with N rows (empty columns if there are no auxiliary signals), and
        the unit of each column ("" for markers and flags, None if unknown).
    """
    parser_units = {"draeger": MEDIBUS_UNITS, "timpel": TIMPEL_AUX_UNITS}.get(
        raw.vendor, {}
    )
    columns, units = {}, {}
    for name, values in (raw.aux_signals or {}).items():
        values = np.asarray(values)
        if raw.vendor == "draeger":
            values = replace_draeger_sentinels(values)
        values = values.astype(float)
        values[~np.isfinite(values)] = np.nan
        if not np.isnan(values).all():
            units[name], factor = common_unit(name, parser_units.get(name))
            columns[name] = values * factor

    if raw.vendor == "draeger":
        for name in DRAEGER_MARKERS:
            if name in raw.frames.dtype.names:
                columns[name] = np.asarray(raw.frames[name], dtype=int)
                units[name] = ""

    return pd.DataFrame(columns, index=pd.RangeIndex(raw.n_frames)), units
