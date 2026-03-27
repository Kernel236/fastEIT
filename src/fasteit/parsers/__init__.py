"""
Parsers for EIT file formats (Dräger and Timpel).

Implemented:
- BaseParser             — abstract base class (ABC) for all parsers
- DragerBinParser   (.bin) — reconstructed 32×32 images; BASE and EXT frame formats
- DragerAscParser   (.asc) — tabular PulmoVista signal export
- TimpelTabularParser (.csv/.txt/.asc) — Timpel tabular recordings

- DragerEitParser   (.eit) — raw transimpedances (208/frame); calibration pipeline EIDORS/Adler 2016

Reconstruction utilities (optional — requires pyeit):
- reconstruct_greit / build_greit — GREIT image reconstruction from RawImpedanceData
  Import: from fasteit.parsers.draeger.eit.eit_pyeit_bridge import reconstruct_greit

Utilities:
- detect_vendor_and_format — extension/vendor/format auto-detection
- load_data                — high-level entry point; returns a BaseData subclass
- load_many                — parse multiple files at once
- load_folder              — parse all matching files in a folder
"""

from .base import BaseParser
from .detection import (
    FileDetection,
    candidate_specs_from_size,
    detect_bin_format_from_size,
    detect_vendor_and_format,
)
from .draeger import DragerAscParser, DragerBinParser, DragerEitParser
from .errors import (
    AmbiguousFormatError,
    InvalidSliceError,
    ParserError,
    UnsupportedFrameSizeError,
)
from .loader import (
    build_parser_from_detection,
    default_parser_registry,
    load_data,
    load_folder,
    load_many,
)
from .timpel import TimpelTabularParser

__all__ = [
    "BaseParser",
    "ParserError",
    "UnsupportedFrameSizeError",
    "AmbiguousFormatError",
    "InvalidSliceError",
    "FileDetection",
    "DragerAscParser",
    "DragerBinParser",
    "DragerEitParser",
    "TimpelTabularParser",
    "build_parser_from_detection",
    "candidate_specs_from_size",
    "default_parser_registry",
    "detect_bin_format_from_size",
    "detect_vendor_and_format",
    "load_data",
    "load_many",
    "load_folder",
]
