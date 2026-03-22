"""Global binary format registry.

This module is vendor-agnostic by design: it stores known `.bin` frame
layouts. Detection logic lives in ``parsers/detection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fasteit.parsers.draeger.bin.draeger_dtypes import (
    FRAME_BASE_DTYPE,
    FRAME_EXT_DTYPE,
    MEDIBUS_BASE_FIELDS,
    MEDIBUS_EXT_FIELDS,
)


@dataclass(frozen=True)
class FormatSpec:
    """Specification of one known `.bin` frame layout.

    ``vendor`` is the routing key used by ``load_data()`` to pick the right
    parser when the file extension alone is ambiguous. Add a new entry to
    ``BIN_FORMAT_SPECS`` to support additional vendors who provide .bin.
    """

    name: str
    vendor: str
    frame_size_bytes: int
    dtype: np.dtype
    medibus_fields: tuple[str, ...] | None
    has_pressure_pod_fields: bool | None


BIN_FORMAT_SPECS: tuple[FormatSpec, ...] = (
    FormatSpec(
        name="Draeger_base_4358",
        vendor="draeger",
        frame_size_bytes=4358,
        dtype=FRAME_BASE_DTYPE,
        medibus_fields=tuple(name for name, _unit, _is_cont in MEDIBUS_BASE_FIELDS),
        has_pressure_pod_fields=False,
    ),
    FormatSpec(
        name="Draeger_ext_4382",
        vendor="draeger",
        frame_size_bytes=4382,
        dtype=FRAME_EXT_DTYPE,
        medibus_fields=tuple(name for name, _unit, _is_cont in MEDIBUS_EXT_FIELDS),
        has_pressure_pod_fields=True,
    ),
)
