"""BaseData: base dataclass for all fasteit data containers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BaseData:
    """Base container for raw data from any parser.

    Subclasses must set n_frames and duration in __post_init__ once their
    primary data array is available.

    Sampling frequency (fs) is parser-derived from file timestamps. No vendor
    default is hardcoded at this layer.

    Attributes:
        filename:    Path of the source file.
        file_format: Format identifier — "bin", "eit", "asc", "txt", or "x".
        vendor:      Detected device vendor — "draeger" or "timpel".
        fs:          Sampling rate in Hz, estimated by parser from timestamps.
        metadata:    Parsed header information (framerate, date, firmware, etc.).
        n_frames:    Number of frames — set by subclass __post_init__.
        duration:    Recording duration in seconds — set by subclass __post_init__.
    """

    filename: str = ""
    file_format: str = ""
    vendor: str = ""
    fs: float | None = None
    metadata: dict = field(default_factory=dict)
    n_frames: int = field(init=False, default=0)
    duration: float = field(init=False, default=0.0)
