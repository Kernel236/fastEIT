"""Clinical indices of a recording (layer 3), as tables ready for a database or R.

- TidalFeatures     - indices of each breath of a recording, and their summary
- PeepTrialFeatures - indices of each breath and of each PEEP step of a PEEP trial

Each object holds everything: its tables, and the objects of the layers before it
(``features.trial.recording.breaths``...). Each value is stored once: the tables of
layer 3 have the keys (``breath_id``, ``step``, ``peep``, ``region``) and what layer 3
computes; what layers 2 and 2b measured (times, quality, pressures, steps) stays in
their own tables, joined on the keys. The tables are "long" by region (one row per
breath, or step, and region), so that they can be joined with the clinical data
(patient, position, side) at the time of the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fasteit.models.eit_recording import EITRecording
from fasteit.models.peep_trial import PeepTrial


# eq=False: compared by identity, not table by table.
@dataclass(frozen=True, eq=False)
class TidalFeatures:
    """Clinical indices of a recording breathing at one setting (output of
    ``tidal_features()``). Do not modify the DataFrames in place: make a copy.

    Attributes:
        recording: The preprocessed recording (layer 2).
        breaths:   One row per breath of the recording: ``breath_id``,
                   ``in_phase``, ``respiratory_rate``, ``eeli``, ``tiv``, ``gi``,
                   ``cov_vd``, ``cov_rl``, ``tidal_volume_used`` (typed or
                   measured) and its ``tidal_volume_source``, ``compliance`` (EIT
                   units per cmH2O), ``compliance_ml`` (mL/cmH2O). Times, quality and
                   pressures: ``recording.breaths``.
        regions:   One row per breath and region: ``breath_id``, ``region``,
                   ``eeli``, ``tiv``, ``share``, ``tiv_ml``, ``compliance``,
                   ``compliance_ml``.
        summary:   One row per region (``"global"`` for the whole lung) and index
                   (column ``feature``, with PEEP, plateau and driving pressure):
                   ``median``, ``q1``, ``q3`` over the valid breaths of the phase,
                   and ``n_breaths`` with a value.
        log:       What was computed, with warnings.
    """

    recording: EITRecording
    breaths: pd.DataFrame
    regions: pd.DataFrame
    summary: pd.DataFrame
    log: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class PeepTrialFeatures:
    """Clinical indices of a PEEP trial (output of ``peep_trial_features()``). Do not
    modify the DataFrames in place: make a copy.

    Attributes:
        trial:        The segmented PEEP trial (layer 2b).
        breaths:      One row per breath, as in ``TidalFeatures`` (without
                      ``in_phase``). Step and analysed breaths: ``trial.breaths``.
        regions:      One row per breath and region, as in ``TidalFeatures``.
        steps:        One row per PEEP step (``step``, ``peep``), from its analysed
                      breaths: medians of the breath indices, compliance in EIT
                      units and in mL/cmH2O (and as % of the best step), Costa
                      collapse and overdistension, and the changes vs the previous
                      and the first step. Driving pressure, tidal volume, breaths
                      analysed: ``trial.steps``.
        step_regions: One row per step and region: ``eeli``, ``tiv``, ``share``,
                      ``tiv_ml``, ``compliance``, ``compliance_ml``.
        summary:      One row per region and feature (``region``, ``feature``,
                      ``value``): for ``"global"`` the crossing PEEP of collapse and
                      overdistension, the PEEP of the best compliance in EIT units and
                      in mL/cmH2O, the two PEEP of the R/I; for every region (global
                      included) the recruitment-to-inflation ratio ``ri`` and the
                      ``recruited_volume_ml``.
        log:          What was computed, with warnings.
    """

    trial: PeepTrial
    breaths: pd.DataFrame
    regions: pd.DataFrame
    steps: pd.DataFrame
    step_regions: pd.DataFrame
    summary: pd.DataFrame
    log: tuple[str, ...]
