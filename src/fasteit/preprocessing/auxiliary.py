"""Auxiliary signals recorded with the images: pressures, flow, volume, device markers.

The parsers keep them as they are in the file (``raw.aux_signals``): for Draeger
``.bin`` the channels that were not available hold "no data" values (-1000, or a
bit pattern read as about -3.4e38) instead of NaN. Here they are cleaned once, so
the next layers read every vendor in the same way.

The signals are not filtered and not filled: a missing value stays NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.draeger.bin.bin_utils import replace_draeger_sentinels

# Draeger per-frame device fields kept with the auxiliary signals.
DRAEGER_MARKERS = ("min_max_flag", "event_marker")


def auxiliary_signals(raw: ReconstructedFrameData) -> pd.DataFrame:
    """One column per auxiliary channel that has data, one row per frame.

    - Draeger "no data" values become NaN (Timpel channels are already NaN from the
      parser);
    - channels without any value are left out (e.g. Medibus when no ventilator was
      connected);
    - Draeger ``min_max_flag`` (+1 end of inspiration, -1 end of expiration, 0
      neither) and ``event_marker`` are added; Timpel ``min_flag``, ``max_flag`` and
      ``qrs_flag`` are already among its channels.

    Rows are frame numbers, the same as ``start_idx``, ``peak_idx`` and ``end_idx``
    of the breath table. Units are those of the parser (Draeger:
    ``fasteit.parsers.draeger.bin.draeger_dtypes``; Timpel: ``timpel_dtypes``).

    Args:
        raw: Parsed recording; never modified.

    Returns:
        DataFrame with N rows (empty columns if there are no auxiliary signals).
    """
    columns = {}
    for name, values in (raw.aux_signals or {}).items():
        values = np.asarray(values)
        if raw.vendor == "draeger":
            values = replace_draeger_sentinels(values)
        values = values.astype(float)
        values[~np.isfinite(values)] = np.nan
        if not np.isnan(values).all():
            columns[name] = values

    if raw.vendor == "draeger":
        for name in DRAEGER_MARKERS:
            if name in raw.frames.dtype.names:
                columns[name] = np.asarray(raw.frames[name], dtype=int)

    return pd.DataFrame(columns, index=pd.RangeIndex(raw.n_frames))
