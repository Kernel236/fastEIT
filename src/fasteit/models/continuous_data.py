"""ContinuousSignalData: container for signal-table EIT exports (no pixel matrices).

Produced by: DragerAscParser (.asc), and future vendors with signal-only exports.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base_data import BaseData


@dataclass
class ContinuousSignalData(BaseData):
    """Container for time-aligned continuous signal tables.

    For files whose primary artifact is a table of sampled signals over time,
    with no pixel matrices.

    Dräger `.asc` is the canonical source. Future vendors producing signal-only
    exports (no pixel matrices) should map here too.

    Attributes:
        table: Parsed tabular data as a pandas DataFrame.
    """

    table: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.table is not None:
            self.n_frames = int(len(self.table))
            self.duration = self.n_frames / self.fs if self.fs else 0.0
