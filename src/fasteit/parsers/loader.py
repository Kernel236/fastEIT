"""High-level parser routing entry points.

Use ``load_data()`` as single public entry point:

1. auto-detect extension/vendor/format for supported vendor and extension.
2. build parser from registry
3. validate+parse via ``parse_safe``
4. attach detection metadata to output

For multiple files use ``load_many()`` or ``load_folder()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from fasteit.models.base_data import BaseData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.draeger import DragerAscParser, DragerBinParser, DragerEitParser

from .detection import FileDetection, detect_vendor_and_format

ParserFactory = Callable[[], BaseParser]

# ---------------------------------------------------------------------------
# Vendor alias normalisation
# ---------------------------------------------------------------------------
# Maps common spelling variants → canonical lowercase vendor identifier.
# Applied before registry lookup so that typos or accented spellings never
# cause a "no parser registered" error.

_VENDOR_ALIASES: dict[str, str] = {
    "draeger": "draeger",
    "drager": "draeger",  # missing 'a' — common typo
    "dräger": "draeger",  # umlaut form
    "timpel": "timpel",
}


def _normalize_vendor(vendor: str) -> str:
    """Return canonical vendor identifier, resolving known aliases."""
    normalised = vendor.strip().lower()
    return _VENDOR_ALIASES.get(normalised, normalised)


def default_parser_registry() -> dict[tuple[str, str], ParserFactory]:
    """Return default (vendor, extension) -> parser factory mapping.

    Keys use canonical lowercase vendor and suffix with dot.
    Timpel entries are added only when the scaffold module is available.
    """
    registry: dict[tuple[str, str], ParserFactory] = {
        ("draeger", ".bin"): lambda: DragerBinParser(),
        ("draeger", ".eit"): lambda: DragerEitParser(),
        ("draeger", ".asc"): lambda: DragerAscParser(),
        ("draeger", ".txt"): lambda: DragerAscParser(),
        ("draeger", ".csv"): lambda: DragerAscParser(),
    }

    try:
        from fasteit.parsers.timpel import (
            TimpelTabularParser as _Timpel,  # noqa: PLC0415
        )

        registry.update(
            {
                ("timpel", ".csv"): lambda: _Timpel(),
                ("timpel", ".txt"): lambda: _Timpel(),
                ("timpel", ".asc"): lambda: _Timpel(),
            }
        )
    except ImportError:
        pass  # Timpel scaffold not yet available (gitignored)

    return registry


def build_parser_from_detection(
    detection: FileDetection,
    *,
    registry: dict[tuple[str, str], ParserFactory] | None = None,
) -> BaseParser:
    """Instantiate parser for an auto-detection payload.

    Vendor string is normalised before lookup so that spelling variants
    (``"drager"``, ``"Dräger"``, ``"DRAEGER"``) all resolve correctly.

    Raises:
        NotImplementedError: when no parser is registered for detected key.
    """
    parser_registry = default_parser_registry() if registry is None else registry
    vendor = _normalize_vendor(detection.vendor)
    key = (vendor, detection.extension.lower())

    if key not in parser_registry:
        available = ", ".join(f"{v}:{e}" for (v, e) in sorted(parser_registry))
        raise NotImplementedError(
            "No parser registered for detected key "
            f"vendor='{detection.vendor}', extension='{detection.extension}'. "
            f"Available keys: [{available}]"
        )

    return parser_registry[key]()


def load_data(
    path: Path,
    *,
    registry: dict[tuple[str, str], ParserFactory] | None = None,
) -> BaseData:
    """Single entry point: detect format/vendor then parse with matching parser."""
    path = Path(path)
    detection = detect_vendor_and_format(path)
    parser = build_parser_from_detection(detection, registry=registry)

    data = parser.parse_safe(path)
    data.vendor = _normalize_vendor(detection.vendor)
    data.metadata.setdefault("detected_vendor", data.vendor)
    data.metadata.setdefault("detected_extension", detection.extension)
    if detection.bin_format is not None:
        data.metadata.setdefault("detected_bin_format", detection.bin_format.name)
    return data


def load_many(
    paths: Iterable[Path | str],
    *,
    registry: dict[tuple[str, str], ParserFactory] | None = None,
) -> list[BaseData]:
    """Parse multiple files and return a list of data containers.

    Each file is parsed independently with ``load_data()``.  Files are
    returned in the same order as ``paths``.

    Args:
        paths:    Iterable of file paths (strings or Path objects).
        registry: Optional custom parser registry; forwarded to ``load_data()``.

    Returns:
        List of ``BaseData`` subclass instances, one per input file.

    Example::

        from fasteit.parsers.loader import load_many

        recordings = load_many(["patient01.bin", "patient02.bin"])
        for rec in recordings:
            print(rec.filename, rec.n_frames, rec.fs)
    """
    return [load_data(Path(p), registry=registry) for p in paths]


def load_folder(
    folder: Path | str,
    pattern: str = "*",
    *,
    registry: dict[tuple[str, str], ParserFactory] | None = None,
) -> list[BaseData]:
    """Parse all matching files in a folder and return a list of data containers.

    Files are sorted by name before parsing.  Use ``pattern`` to filter by
    extension or naming convention (standard ``glob`` syntax).

    Args:
        folder:   Directory to scan.
        pattern:  Glob pattern relative to ``folder``.  Default ``"*"`` matches
                  every file directly inside the folder (non-recursive).
                  Use ``"**/*.bin"`` for recursive search.
        registry: Optional custom parser registry; forwarded to ``load_data()``.

    Returns:
        List of ``BaseData`` subclass instances for all matched files that
        could be parsed.  Files that raise an error are skipped with a warning
        printed to stderr.

    Example::

        from fasteit.parsers.loader import load_folder

        # All .bin files in a folder
        recordings = load_folder("/data/patient01/", pattern="*.bin")

        # Mixed folder — all supported formats
        recordings = load_folder("/data/patient01/")
    """
    import sys

    folder = Path(folder)
    paths = sorted(folder.glob(pattern))
    results: list[BaseData] = []
    for p in paths:
        if not p.is_file():
            continue
        try:
            results.append(load_data(p, registry=registry))
        except Exception as exc:  # noqa: BLE001
            print(f"[load_folder] skipping '{p.name}': {exc}", file=sys.stderr)
    return results
