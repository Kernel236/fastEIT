"""PeepTrial: a preprocessed PEEP trial split into PEEP steps (layer 2b)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fasteit.config import SegmentationConfig
from fasteit.models.eit_recording import EITRecording


# eq=False: two trials are compared by identity, not table by table.
@dataclass(frozen=True, eq=False)
class PeepTrial:
    """A PEEP trial ready for layer 3: PEEP steps and the breaths analysed in each.

    Output of ``segment_peep_trial()``. To change a parameter or a step placed by hand,
    call it again. Do not modify the DataFrames in place: make a copy.

    Attributes:
        recording:        The preprocessed recording (layer 2).
        config:           Parameters used.
        pressure_channel: Column of ``recording.aux`` the pressures were read from, or
                          None if there was no pressure signal.
        pressure_unit:    ``"mbar"`` (Draeger) or ``"cmH2O"`` (Timpel); None if unknown.
        source:           Where the steps come from: ``"pressure"`` or ``"manual"``.
        steps:            One row per step: ``step``, ``peep`` (level, as measured or as
                          typed), ``start_time``, ``end_time``, ``n_breaths``,
                          ``n_analysed``, ``enough`` (at least the minimum), ``plateau``
                          and ``driving_pressure`` (medians of the analysed breaths, or
                          the plateau typed with a step placed by hand),
                          ``n_driving_pressure`` and ``enough_driving_pressure``.
        breaths:          The breath table of the recording plus ``peep``, ``plateau``,
                          ``driving_pressure`` (units of the recording), ``peep_step``
                          (-1 outside every step) and ``analysed``.
        log:              What the segmentation did, with warnings.
    """

    recording: EITRecording
    config: SegmentationConfig
    pressure_channel: str | None
    pressure_unit: str | None
    source: str
    steps: pd.DataFrame
    breaths: pd.DataFrame
    log: tuple[str, ...]

    @property
    def analysed_breaths(self) -> pd.DataFrame:
        """The breaths analysed in each step."""
        return self.breaths[self.breaths.analysed]
