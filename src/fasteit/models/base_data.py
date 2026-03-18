"""BaseData: base dataclass for all fasteit data containers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseData:
    """Base container for raw data from any parser.

    Subclasses must set n_frames and duration in __post_init__ once their
    primary data array is available.

    The fs default (50.0 Hz) matches the PulmoVista 500 acquisition rate for
    all file formats (.bin, .eit, .txt). Parsers always pass fs explicitly
    when constructing a subclass instance — the default is a fallback for
    manually constructed objects (e.g. in tests).

    Attributes:
        filename:    Path of the source file.
        file_format: Format identifier — "bin", "eit", or "csv".
        fs:          Sampling rate in Hz. Default 50.0 (PulmoVista 500 standard).
        metadata:    Parsed header information (framerate, date, firmware, etc.).
        n_frames:    Number of frames — set by subclass __post_init__.
        duration:    Recording duration in seconds — set by subclass __post_init__.
    """

    filename: str = ""
    file_format: str = ""
    fs: float = 50.0  # Hz — Dräger PulmoVista 500 acquisition rate
    metadata: dict = field(default_factory=dict)
    n_frames: int = field(init=False, default=0)
    duration: float = field(init=False, default=0.0)
