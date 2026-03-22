"""Shared parser exceptions across vendors and file formats."""


class ParserError(Exception):
    """Base exception for all fasteit parser failures."""


class UnsupportedFrameSizeError(ParserError):
    """File size is not divisible by any known frame size."""


class AmbiguousFormatError(ParserError):
    """More than one format candidate remains after detection."""


class InvalidSliceError(ParserError):
    """Requested first_frame/max_frames window is invalid."""
