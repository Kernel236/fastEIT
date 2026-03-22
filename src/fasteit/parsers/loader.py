"""High-level parser routing entry points.

Use ``load_data()`` as single public entry point:

1. auto-detect extension/vendor/format for supported vendor and extension.
2. build parser from registry
3. validate+parse via ``parse_safe``
4. attach detection metadata to output
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fasteit.models.base_data import BaseData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.draeger import DragerAscParser, DragerBinParser, DragerEitParser
from fasteit.parsers.timpel import TimpelTabularParser

from .detection import FileDetection, detect_vendor_and_format

ParserFactory = Callable[[], BaseParser]


def default_parser_registry() -> dict[tuple[str, str], ParserFactory]:
    """Return default (vendor, extension) -> parser factory mapping.

    Keys use lowercase vendor and suffix with dot, e.g. ``("draeger", ".bin")``.
    Add future vendors/parsers by extending this dictionary.
    """
    return {
        ("draeger", ".bin"): lambda: DragerBinParser(),
        ("draeger", ".eit"): lambda: DragerEitParser(),
        ("draeger", ".asc"): lambda: DragerAscParser(),
        ("draeger", ".txt"): lambda: DragerAscParser(),
        ("draeger", ".csv"): lambda: DragerAscParser(),
        ("timpel", ".csv"): lambda: TimpelTabularParser(),
        ("timpel", ".txt"): lambda: TimpelTabularParser(),
        ("timpel", ".asc"): lambda: TimpelTabularParser(),
    }


def build_parser_from_detection(
    detection: FileDetection,
    *,
    registry: dict[tuple[str, str], ParserFactory] | None = None,
) -> BaseParser:
    """Instantiate parser for an auto-detection payload.

    Raises:
        NotImplementedError: when no parser is registered for detected key.
    """
    parser_registry = default_parser_registry() if registry is None else registry
    key = (detection.vendor.lower(), detection.extension.lower())

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
    """Single entry point: detect format/vendor then parse with matching parser.
    """
    path = Path(path)
    detection = detect_vendor_and_format(path)
    parser = build_parser_from_detection(
        detection,
        registry=registry,
    )

    data = parser.parse_safe(path)
    data.vendor = detection.vendor
    data.metadata.setdefault("detected_vendor", detection.vendor)
    data.metadata.setdefault("detected_extension", detection.extension)
    if detection.bin_format is not None:
        data.metadata.setdefault("detected_bin_format", detection.bin_format.name)
    return data
