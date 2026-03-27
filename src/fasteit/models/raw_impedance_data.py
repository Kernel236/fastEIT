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
        measurements: Calibrated transimpedance array, shape (N_frames, 208).
                      Values are gain-corrected (e.g. ``vv = FT_A*trans_A - FT_B*trans_B``
                      for Dräger), not raw ADC counts and not yet image-reconstructed.
                      208 = 16 electrodes × 13 measurement pairs
                      (adjacent drive, standard Dräger pattern).
        aux_signals:  Optional dict of auxiliary waveforms keyed by signal
                      name, e.g. ``{"timestamp": ..., "frame_counter": ...,
                      "medibus_Paw": ...}``. Each value is a 1-D array of
                      length N_frames.
    """

    measurements: np.ndarray | None = None
    aux_signals: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.measurements is not None:
            self.n_frames = len(self.measurements)
            self.duration = self.n_frames / self.fs if self.fs else 0.0
