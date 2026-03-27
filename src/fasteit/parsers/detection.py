"""File-type and vendor detection templates for parser routing.

This module centralizes auto-detection used by ``load_data()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bin_formats import BIN_FORMAT_SPECS, FormatSpec
from .errors import AmbiguousFormatError, UnsupportedFrameSizeError
from .header_formats import HEADER_FORMAT_SPECS


@dataclass(frozen=True)
class FileDetection:
    """Auto-detection result consumed by parser factory/routing.

    Attributes:
        path: Input file path.
        extension: Lowercased suffix including dot, e.g. ``.bin``.
        vendor: Vendor identifier, e.g. ``draeger``.
        bin_format: Selected binary frame layout for ``.bin`` files.
    """

    path: Path
    extension: str
    vendor: str
    bin_format: FormatSpec | None = None


def detect_vendor_and_format(path: Path) -> FileDetection:
    """Detect extension, vendor, and binary frame format when relevant.

    Args:
        path: Input file path.

    Returns:
        ``FileDetection`` routing payload for parser selection.
    """
    path = Path(path)
    extension = path.suffix.lower()

    if extension == ".bin":
        spec = detect_bin_format_from_size(path)
        return FileDetection(
            path=path,
            extension=extension,
            vendor=spec.vendor,
            bin_format=spec,
        )

    if extension == ".eit":
        vendor = detect_vendor_from_eit_header(path)
        return FileDetection(path=path, extension=extension, vendor=vendor)

    if extension in {".csv", ".txt", ".asc"}:
        vendor = detect_vendor_from_tabular(path)
        return FileDetection(path=path, extension=extension, vendor=vendor)

    raise ValueError(
        f"Unsupported extension '{extension}' for file '{path}'. "
        "Supported templates: .bin, .eit, .csv, .txt, .asc."
    )


def candidate_specs_from_size(file_size: int) -> list[FormatSpec]:
    """Return binary format specs whose frame size divides file size exactly."""
    return [spec for spec in BIN_FORMAT_SPECS if file_size % spec.frame_size_bytes == 0]


def detect_bin_format_from_size(path: Path) -> FormatSpec:
    """Detect binary frame format from headerless file size divisibility."""
    path = Path(path)
    file_size = path.stat().st_size
    candidates = candidate_specs_from_size(file_size)

    if not candidates:
        supported = [spec.frame_size_bytes for spec in BIN_FORMAT_SPECS]
        raise UnsupportedFrameSizeError(
            f"Unsupported .bin size {file_size} bytes. Known frame sizes: {supported}."
        )

    if len(candidates) == 1:
        return candidates[0]

    names = [c.name for c in candidates]
    raise AmbiguousFormatError(
        "Ambiguous .bin format by frame-size divisibility. "
        f"Candidates: {names}. This project currently fails fast on ambiguity "
    )


def detect_vendor_from_eit_header(path: Path) -> str:
    """Detect vendor from ``.eit`` file header by matching a known magic string.

    Iterates ``HEADER_FORMAT_SPECS`` in order. For each spec, reads
    ``magic_search_bytes`` bytes and searches for ``magic_string``.
    Returns the vendor of the first match.

    Args:
        path: Path to the ``.eit`` file.

    Returns:
        Vendor identifier string (e.g. ``"draeger"``).

    Raises:
        ValueError: If no registered magic string matches.
        OSError: If the file cannot be read.
    """
    path = Path(path)

    max_read = max(s.magic_search_bytes for s in HEADER_FORMAT_SPECS)
    with path.open("rb") as f:
        raw = f.read(max_read)

    for spec in HEADER_FORMAT_SPECS:
        region = raw[: spec.magic_search_bytes].decode(spec.encoding, errors="replace")
        if spec.magic_string in region:
            return spec.vendor

    known = [s.magic_string for s in HEADER_FORMAT_SPECS]
    raise ValueError(
        f"Could not detect vendor for .eit file '{path}'. "
        f"No registered magic string found. Known: {known}."
    )


def detect_vendor_from_tabular(path: Path) -> str:
    """Detect vendor for CSV/TXT/ASC tabular files.

    Current rules (minimal but production-usable):
    - Drager ASC export: header contains ``---DraegerEIT Software``
      and usually a ``Tidal Variations`` section.
    - Timpel tabular export (heuristic): first non-empty line is a dense
      numeric matrix row with ~1030 values separated by comma/tab/semicolon,
      or explicit ``timpel`` keyword in header text.
    """
    path = Path(path)
    with path.open("r", encoding="latin1", errors="replace") as f:
        first_lines = [next(f, "") for _ in range(40)]

    normalized = "\n".join(first_lines).lower()
    if "draegereit software" in normalized:
        return "draeger"
    if "timpel" in normalized:
        return "timpel"

    first_non_empty = ""
    for line in first_lines:
        stripped = line.strip()
        if stripped:
            first_non_empty = stripped
            break

    if first_non_empty:
        comma_count = first_non_empty.count(",")
        tab_count = first_non_empty.count("\t")
        semi_count = first_non_empty.count(";")
        if max(comma_count, tab_count, semi_count) >= 1029:
            return "timpel"

    raise ValueError(
        f"Could not detect vendor for tabular file '{path}'. "
        "Expected Draeger ASC header or known tabular signature."
    )
