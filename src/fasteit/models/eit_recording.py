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


@dataclass(frozen=True)
class Segment:
    """A time interval of a recording, in seconds from the start of the recording.

    Attributes:
        start:      Start time in seconds.
        end:        End time in seconds (must be greater than ``start``).
        category:   What the interval is. Layer 2 uses ``"exclusion"``; layer 3 will
                    add ``"peep_step"`` and ``"transition"``.
        label:      Why, e.g. ``"artifact"``, ``"movement"``, ``"timing_error"``.
        source:     Who created it: ``"manual"``, ``"device"`` or ``"auto"``.
        parameters: For automatic segments, how they were found. Not used for
                    hashing, so segments can be put in a set.

    Two segments that share an end point, e.g. [0, 10] and [10, 20], touch: a breath
    ending at 10 s overlaps both.
    """

    start: float
    end: float
    category: str = "exclusion"
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

    Immutable: to change a parameter or an exclusion, call ``preprocess()`` again.
    The arrays are read-only and the parsed data in ``raw`` is never modified.

    Attributes:
        raw:           Data from the parser, untouched.
        config:        Parameters used.
        context:       Attributes of the recording given by the user.
        exclusions:    Excluded parts (manual and device), widened by the margin and
                       merged.
        time:          (N,) seconds from the start of the recording.
        rates:         ``respiratory_rate_hz`` from the spectrum (None if it could
                       not be estimated) and ``heart_rate_hz`` (None if not given).
        pixels:        (N, rows, cols) float32, after the cardiac step.
        global_signal: (N,) sum of the pixels inside the lung mask.
        breaths:       One row per breath: times and frames of start, peak and end,
                       quality flags, ``excluded``, ``exclusion_label``,
                       ``is_valid``.
        lung_mask:     (rows, cols) bool, functional lung.
        tiv_map:       (rows, cols) mean TIV of the valid breaths, used for the mask.
        roi_masks:     Regions of interest, each inside the lung mask.
        roi_signals:   (N, n_regions) sum of the pixels of each region.
        log:           What each step did, with fallbacks and warnings.
    """

    raw: ReconstructedFrameData
    config: PreprocessingConfig
    context: RecordingContext
    exclusions: tuple[Segment, ...]
    time: np.ndarray
    rates: dict
    pixels: np.ndarray
    global_signal: np.ndarray
    breaths: pd.DataFrame
    lung_mask: np.ndarray
    tiv_map: np.ndarray
    roi_masks: dict[str, np.ndarray]
    roi_signals: pd.DataFrame
    log: tuple[str, ...]

    def __post_init__(self) -> None:
        read_only = [
            self.time,
            self.pixels,
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
