"""Global registry for header-based EIT file formats.

This module is vendor-agnostic by design: it stores format
specifications used for vendor detection using known header field mapping. Detection
logic lives in ``parsers/detection.py``; format-specific frame parsing lives
in each vendor's parser module.

To add a new vendor file with header + frames:
1. Define a named ``HeaderFormatSpec`` constant (e.g. ``CAREFUSION_EIT_HEADER_SPEC``).
2. Add it to ``HEADER_FORMAT_SPECS``.
3. Write a parser in ``parsers/<vendor>/eit/`` that calls ``get_eit_spec()``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeaderFormatSpec:
    """Specification of one known header-based ``.eit`` file format.

    Covers both *detection* (magic string matching) and *parsing* (header
    field mapping). A parser can retrieve its own spec via ``get_eit_spec()``.

    Attributes:
        name:               Human-readable format identifier.
        vendor:             Vendor routing key (e.g. ``"draeger"``).
        magic_string:       ASCII substring to search for anywhere in the
                            first ``magic_search_bytes`` bytes of the file.
        magic_search_bytes: How many bytes to read from the file start for
                            detection. Keeps detection fast on large files.
        encoding:           Text encoding of the header (e.g. ``"latin-1"``).
        frame_size_bytes:    Size of each binary data frame in bytes.
                             Used by the parser to disambiguate between multiple
                             specs for the same vendor.
        n_electrodes:        Number of electrodes on the electrode belt.
        n_measurements:      Number of transimpedance measurements per frame.
                             For adjacent-drive 16-electrode systems:
                             16 injections × 13 measurements = 208.
    """

    name: str
    vendor: str
    magic_string: str
    magic_search_bytes: int
    encoding: str
    frame_size_bytes: int
    n_electrodes: int
    n_measurements: int


DRAEGER_EIT_HEADER_SPEC = HeaderFormatSpec(
    name="Draeger_EIT_v51",
    vendor="draeger",
    magic_string="Draeger EIT-Software",
    magic_search_bytes=512,  # magic string appears ~byte 51, 512 is safe
    encoding="latin-1",
    frame_size_bytes=5495,
    n_electrodes=16,
    n_measurements=208,  # 16 injections × 13 adjacent measurements
)

# Future: Carefusion .eit format
# CAREFUSION_EIT_HEADER_SPEC = HeaderFormatSpec(
#     name="Carefusion_EIT",
#     vendor="carefusion",
#     magic_string="CareFusion",
#     magic_search_bytes=256,
#     encoding="utf-8",
#     frame_size_bytes=...,
# )

HEADER_FORMAT_SPECS: tuple[HeaderFormatSpec, ...] = (
    DRAEGER_EIT_HEADER_SPEC,
    # CAREFUSION_EIT_HEADER_SPEC,
)


def get_eit_specs(vendor: str) -> list[HeaderFormatSpec]:
    """Return all ``HeaderFormatSpec`` entries registered for *vendor*.

    Multiple specs for the same vendor are valid — e.g. two Dräger firmware
    versions with the same magic string but different frame sizes. The caller
    is responsible for disambiguating by ``frame_size_bytes``.

    Args:
        vendor: Vendor identifier (e.g. ``"draeger"``).

    Returns:
        List of matching ``HeaderFormatSpec`` entries (order preserved from
        ``HEADER_FORMAT_SPECS``). Never empty.

    Raises:
        ValueError: If no spec is registered for the given vendor.
    """
    matches = [s for s in HEADER_FORMAT_SPECS if s.vendor == vendor]
    if not matches:
        registered = [s.vendor for s in HEADER_FORMAT_SPECS]
        raise ValueError(
            f"No HeaderFormatSpec registered for vendor '{vendor}'. "
            f"Registered vendors: {registered}."
        )
    return matches
