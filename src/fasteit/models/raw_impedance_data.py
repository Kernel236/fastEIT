"""RawImpedanceData: container for raw transimpedance measurements.

Produced by: DragerEitParser (.eit), TimpelRawParser (.x).
Primary use case: pyEIT interface for image reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base_data import BaseData


@dataclass
class RawImpedanceData(BaseData):
    """Container for raw EIT transimpedance measurements.

    Stores the 208 transimpedance values per frame as-is from the file,
    before any image reconstruction. The intended consumer is pyEIT.

    Attributes:
        measurements: Raw transimpedance array, shape (N_frames, 208).
                      208 = 16 electrodes × 13 measurement pairs
                      (adjacent drive, standard Dräger/Timpel pattern).
    """

    measurements: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.measurements is not None:
            self.n_frames = len(self.measurements)
            self.duration = self.n_frames / self.fs if self.fs else 0.0
