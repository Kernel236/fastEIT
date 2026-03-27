"""Utility functions for Dräger `.eit` header parsing.

Mirrors the role of ``bin_utils.py`` for the binary parser: low-level
parsing helpers that ``DragerEitParser`` delegates to.
"""

from __future__ import annotations

import numpy as np

# 8-byte separator between ASCII header and binary frame data.
SEPARATOR: bytes = b"**\r\n\r\n\r\n"

# Empirical calibration constants — EIDORS read_draeger_file.m (GPL),
# estimated 2016-04-07 by A. Adler. Understanding only — no code copied.
# Not present in the .eit header; hardcoded from EIDORS best-guess estimates.
#   FT_A, FT_B : transimpedance scaling  (vv = FT_A*trans_A - FT_B*trans_B)
#   FC_CURRENT : ADC counts → Ampere     (injection_current / FC_CURRENT)
#   FV_VOLTAGE : ADC counts → Volt       ((voltage_A - voltage_B) / FV_VOLTAGE)
FT_A: float = 0.00098242
FT_B: float = 0.00019607
FC_CURRENT: float = 194326.3536
FV_VOLTAGE: float = 0.11771

# Mapping of ASCII header field names → (metadata_key, type_converter).
# Fields not listed here are kept verbatim in metadata["_raw_fields"].
HEADER_FIELD_MAP: dict[str, tuple[str, type]] = {
    "Framerate [Hz]": ("fs", float),
    "Date": ("date", str),
    "Time": ("time", str),
    "Gain": ("gain", int),
    "Samples": ("samples_per_period", int),
    "Periods": ("periods", int),
    "Frequency [kHz]": ("frequency_khz", float),
    "Amplitude [uA]": ("amplitude_ua", float),
    "Format": ("format_version_ascii", int),
}


def parse_eit_header(raw: bytes) -> tuple[dict, int]:
    """Parse the Dräger .eit ASCII header from raw file bytes.

    Reads the 12-byte preamble (3 × int32 LE) to locate ``sep_offset``, then
    decodes the ASCII header text and extracts structured metadata using
    ``HEADER_FIELD_MAP``.

    Args:
        raw: Raw bytes from the start of the file, must cover at least
             ``sep_offset + len(SEPARATOR)`` bytes (the full header region).

    Returns:
        A tuple of:
        - ``metadata``: dict with typed fields from ``HEADER_FIELD_MAP``, plus
          ``"format_version"`` (from preamble), ``"binary_start"`` (byte
          offset of first frame), and ``"_raw_fields"`` (all raw key-value
          pairs for debugging).
        - ``binary_start``: byte offset where binary frame data begins
          (= ``sep_offset + len(SEPARATOR)``).

    Raises:
        ValueError: If the preamble is unreadable or ``sep_offset`` is invalid.
    """
    _PREAMBLE_DTYPE = np.dtype("<i4")
    _PREAMBLE_SIZE = 3 * _PREAMBLE_DTYPE.itemsize

    if len(raw) < _PREAMBLE_SIZE:
        raise ValueError(
            f"File too small: need at least {_PREAMBLE_SIZE} bytes for preamble, "
            f"got {len(raw)}."
        )

    preamble = np.frombuffer(raw[:_PREAMBLE_SIZE], dtype=_PREAMBLE_DTYPE)
    fmt_version, sep_offset, _ = int(preamble[0]), int(preamble[1]), int(preamble[2])

    if sep_offset <= _PREAMBLE_SIZE or sep_offset + len(SEPARATOR) > len(raw):
        raise ValueError(
            f"Invalid sep_offset={sep_offset}: out of range for buffer of "
            f"{len(raw)} bytes. File may be truncated or not a valid .eit."
        )

    header_text = raw[_PREAMBLE_SIZE:sep_offset].decode("latin-1", errors="replace")

    metadata: dict = {"format_version": fmt_version}
    raw_fields: dict[str, str] = {}

    for line in header_text.split("\r\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        raw_fields[key] = value

        if key in HEADER_FIELD_MAP:
            out_key, converter = HEADER_FIELD_MAP[key]
            try:
                metadata[out_key] = converter(value)
            except (ValueError, TypeError):
                metadata[out_key] = value  # keep as string on conversion failure

    metadata["_raw_fields"] = raw_fields

    binary_start = sep_offset + len(SEPARATOR)
    metadata["binary_start"] = binary_start

    return metadata, binary_start
