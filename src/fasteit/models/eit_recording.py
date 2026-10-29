"""Containers for a preprocessed recording (layer 2).

- Segment          - a time interval of a recording (exclusion, later PEEP step)
- RecordingContext - attributes of the whole recording (patient, position, protocol)
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
        parameters: For automatic segments, how they were found.
    """

    start: float
    end: float
    category: str = "exclusion"
    label: str = ""
    source: str = "manual"
    parameters: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
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
