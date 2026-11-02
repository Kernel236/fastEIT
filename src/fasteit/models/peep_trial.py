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
        source:           Where the steps come from: ``"pressure"`` or ``"manual"``.
        steps:            One row per step: ``step``, ``peep`` (level, as measured or as
                          typed, cmH2O), ``peep_set`` (the nearest PEEP level of the
                          protocol, ``SegmentationConfig.peep_levels``), ``start_time``, ``end_time``, ``n_breaths``,
                          ``n_analysed``, ``enough`` (at least the minimum), ``plateau``
                          and ``driving_pressure`` with ``driving_pressure_source``:
                          ``"breaths"`` (median of the analysed breaths, continuous
                          pause), ``"hold"`` (median of the breaths of the step with an
                          inspiratory hold) or ``"typed"`` (step placed by hand);
                          ``n_driving_pressure`` and ``enough_driving_pressure``;
                          ``tidal_volume`` (mL) with ``tidal_volume_source``:
                          ``"typed"`` (for the step), ``"typed_recording"``
                          (``RecordingContext.tidal_volume_ml``) or ``"breaths"``
                          (median of the analysed breaths, ventilator curves).
        breaths:          The breath table of the recording (with ``peep``,
                          ``plateau`` and ``driving_pressure`` in cmH2O and
                          ``tidal_volume`` in mL, read in layer 2) plus ``peep_step``
                          (-1 outside every step) and ``analysed``.
        log:              What the segmentation did, with warnings.
    """

    recording: EITRecording
    config: SegmentationConfig
    source: str
    steps: pd.DataFrame
    breaths: pd.DataFrame
    log: tuple[str, ...]

    @property
    def analysed_breaths(self) -> pd.DataFrame:
        """The breaths analysed in each step."""
        return self.breaths[self.breaths.analysed]
