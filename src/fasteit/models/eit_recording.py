"""Containers for a preprocessed recording (layer 2).

- Segment          - a time interval of a recording (exclusion, later PEEP step)
- RecordingContext - attributes of the whole recording (patient, position, protocol)
- EITRecording     - the recording ready for analysis, output of ``preprocess()``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fasteit.config import PreprocessingConfig
from fasteit.models.reconstructed_data import ReconstructedFrameData

# Segment categories.
EXCLUSION = "exclusion"  # a part not analysed (artifact, maneuver); layer 2
PEEP_STEP = "peep_step"  # a PEEP step placed by hand, with parameters["peep"]; layer 2b
PHASE = "phase"  # a part with its own protocol, label "tidal" or "peep_trial"; layer 2b


@dataclass(frozen=True)
class Segment:
    """A time interval of a recording, in seconds from the start of the recording.

    Attributes:
        start:      Start time in seconds.
        end:        End time in seconds (must be greater than ``start``).
        category:   What the interval is: ``"exclusion"`` (layer 2), ``"peep_step"``
                    or ``"phase"`` (layer 2b).
        label:      Why, e.g. ``"artifact"``, ``"movement"``, ``"timing_error"``.
        source:     Who created it: ``"manual"``, ``"device"`` or ``"auto"``.
        parameters: For automatic segments, how they were found. Not used for
                    hashing, so segments can be put in a set.

    Two segments that share an end point, e.g. [0, 10] and [10, 20], touch: a breath
    ending at 10 s overlaps both.
    """

    start: float
    end: float
    category: str = EXCLUSION
    label: str = ""
    source: str = "manual"
    parameters: dict = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError(
                f"Segment times must be finite, got {self.start}, {self.end}"
            )
        if self.start < 0:
            raise ValueError(f"Segment start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"Segment end ({self.end}) must be after start ({self.start})"
            )

    def overlaps(self, start: float, end: float) -> bool:
        """Return True if the interval [start, end] touches this segment."""
        return start <= self.end and end >= self.start


@dataclass(frozen=True)
class RecordingContext:
    """Attributes of the whole recording, given by the user.

    Attributes:
        patient_id: Anonymised patient identifier.
        position:   Patient position during the recording, e.g. ``"supine"``, ``"prone"``.
        protocol:   ``"tidal"`` or ``"peep_trial"``; decides what layers 3-4 compute.
        interface:  Ventilation interface, e.g. ``"helmet"``, ``"face_mask"``.
        notes:      Free text.
    """

    patient_id: str = ""
    position: str | None = None
    protocol: str = "tidal"
    interface: str | None = None
    notes: str = ""


# eq=False: two recordings are compared by identity, not array by array.
@dataclass(frozen=True, eq=False)
class EITRecording:
    """A recording ready for breath-by-breath analysis (output of ``preprocess()``).

    To change a parameter or an exclusion, call ``preprocess()`` again. The fields
    cannot be reassigned, the config is frozen, the arrays are read-only and the
    parsed data in ``raw`` is never modified. The DataFrames and dicts can still be
    changed in place by Python: do not modify them, make a copy.

    All signals are in arbitrary impedance units (EELI is relative, not comparable
    between recordings). Breath frames (``start_idx``, ``peak_idx``, ``end_idx``)
    are extrema of ``detection_signal``, the signal they were found on.

    Attributes:
        raw:           Data from the parser, untouched.
        config:        Parameters used.
        context:       Attributes of the recording given by the user.
        exclusions:    Excluded parts (manual and device), widened by the margin and
                       merged.
        time:          (N,) seconds from the start of the recording.
        rates:         ``respiratory_rate_hz`` from the spectrum (None if it could
                       not be estimated) and ``heart_rate_hz`` as entered by the
                       user (None if not given).
        cardiac_filter: Filter actually applied: ``"mdn"`` (cardiac component
                       removed) or ``"moving_average"`` (not removed).
        pixels:        (N, rows, cols) float32, after the cardiac step; NaN for pixels
                       without data in the whole recording.
        image_mask:    (rows, cols) bool, pixels with a signal (the image inside the
                       belt outline); wider than the lung, e.g. for silent spaces.
        detection_signal: (N,) sum of all pixels after the cardiac step, used for
                       the breath detection and the quality flags.
        global_signal: (N,) sum of the pixels inside the lung mask.
        breaths:       One row per breath: times and frames of start, peak and end,
                       quality flags, ``excluded``, ``exclusion_label``,
                       ``is_valid``.
        lung_mask:     (rows, cols) bool, ventilated lung (not the anatomical lung).
        tiv_map:       (rows, cols) mean TIV of the valid breaths (first and last
                       breath excluded), used for the mask.
        roi_masks:     Regions of interest, each inside the lung mask.
        roi_signals:   (N, n_regions) sum of the pixels of each region.
        aux:           (N, n_channels) auxiliary signals with data (pressures, flow,
                       volume, device breath markers), "no data" values as NaN, not
                       filtered; rows are frame numbers like the breath indices.
        log:           What each step did, with fallbacks and warnings.
    """

    raw: ReconstructedFrameData
    config: PreprocessingConfig
    context: RecordingContext
    exclusions: tuple[Segment, ...]
    time: np.ndarray
    rates: dict[str, float | None]
    cardiac_filter: str
    pixels: np.ndarray
    image_mask: np.ndarray
    detection_signal: np.ndarray
    global_signal: np.ndarray
    breaths: pd.DataFrame
    lung_mask: np.ndarray
    tiv_map: np.ndarray
    roi_masks: dict[str, np.ndarray]
    roi_signals: pd.DataFrame
    aux: pd.DataFrame
    log: tuple[str, ...]

    def __post_init__(self) -> None:
        read_only = [
            self.time,
            self.pixels,
            self.image_mask,
            self.detection_signal,
            self.global_signal,
            self.lung_mask,
            self.tiv_map,
            *self.roi_masks.values(),
        ]
        for array in read_only:
            array.flags.writeable = False

    @property
    def fs(self) -> float:
        """Sampling frequency in Hz."""
        return self.raw.fs

    @property
    def valid_breaths(self) -> pd.DataFrame:
        """The breaths without quality flags and outside the exclusions."""
        return self.breaths[self.breaths.is_valid]
