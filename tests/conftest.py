"""Shared pytest fixtures for fasteit tests.

Fixtures here are available to all test modules automatically.
"""

from __future__ import annotations

import numpy as np
import pytest

from fasteit.parsers.draeger.bin.draeger_dtypes import FRAME_BASE_DTYPE

_FS_HZ: float = 50.0
_DT_DAY: float = 1.0 / (_FS_HZ * 86400.0)  # one frame in Dräger fraction-of-day units


@pytest.fixture
def bin_3frames(tmp_path):
    """3-frame Dräger BASE .bin file with known pixel values.

    Frame i has all pixels = float(i + 1):
        frame 0 → pixels all 1.0
        frame 1 → pixels all 2.0
        frame 2 → pixels all 3.0

    Timestamps at 50 Hz in Dräger fraction-of-day format.
    File size: 3 × 4358 = 13 074 bytes (valid BASE format).

    Returns:
        Path to the synthetic .bin file.
    """
    frames = np.zeros(3, dtype=FRAME_BASE_DTYPE)
    for i in range(3):
        frames[i]["ts"] = i * _DT_DAY
        frames[i]["pixels"][:] = float(i + 1)
    path = tmp_path / "synthetic_3f.bin"
    frames.tofile(path)
    return path
