"""Template tests for auto-detection and parser routing flow."""

from pathlib import Path

import pytest

from fasteit.parsers.detection import detect_vendor_and_format
from fasteit.parsers.draeger import DragerBinParser
from fasteit.parsers.loader import build_parser_from_detection


def test_detect_vendor_and_format_bin_from_frame_size(tmp_path: Path):
    """`.bin` vendor detection uses BIN_FORMAT_SPECS via frame-size divisibility."""
    # 4358 bytes -> base_4358 candidate for draeger registry entry.
    p = tmp_path / "sample.bin"
    p.write_bytes(b"\x00" * 4358)

    det = detect_vendor_and_format(p)

    assert det.extension == ".bin"
    assert det.vendor == "draeger"
    assert det.bin_format is not None
    assert det.bin_format.name == "Draeger_base_4358"


def test_build_parser_from_detection_bin_returns_draeger_bin_parser(tmp_path: Path):
    p = tmp_path / "sample.bin"
    p.write_bytes(b"\x00" * 4358)
    det = detect_vendor_and_format(p)

    parser = build_parser_from_detection(det)

    assert isinstance(parser, DragerBinParser)


def test_build_parser_from_detection_unknown_key_raises():
    from fasteit.parsers.detection import FileDetection

    det = FileDetection(
        path=Path("dummy.xyz"),
        extension=".xyz",
        vendor="unknown_vendor",
        bin_format=None,
    )

    with pytest.raises(NotImplementedError, match="No parser registered"):
        build_parser_from_detection(det)


def test_to_detect_lowcase_functionality():
    from fasteit.parsers.detection import FileDetection

    uppercase_vendor = FileDetection(
        path=Path("dummy.BIN"),
        extension=".BIN",
        vendor="DRAEGER",
        bin_format=None,
    )

    parser = build_parser_from_detection(uppercase_vendor)
    assert isinstance(parser, DragerBinParser)
