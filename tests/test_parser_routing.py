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


def test_detect_vendor_and_format_unsupported_extension_raises(tmp_path: Path):
    p = tmp_path / "file.xyz"
    p.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="Unsupported extension"):
        detect_vendor_and_format(p)


def test_detect_vendor_and_format_ambiguous_bin_raises(tmp_path: Path):
    # lcm(4358, 4382) = 4358 * 4382 / gcd(4358, 4382) = 9_543_478 bytes
    # Compute the actual lcm
    import math

    from fasteit.parsers.errors import AmbiguousFormatError
    lcm_size = math.lcm(4358, 4382)
    p = tmp_path / "ambiguous.bin"
    p.write_bytes(b"\x00" * lcm_size)
    with pytest.raises(AmbiguousFormatError):
        detect_vendor_and_format(p)


def test_detect_vendor_from_tabular_timpel_keyword(tmp_path: Path):
    from fasteit.parsers.detection import detect_vendor_from_tabular

    p = tmp_path / "timpel_data.txt"
    p.write_text("Timpel measurement export\nsome data\n", encoding="latin1")
    assert detect_vendor_from_tabular(p) == "timpel"


def test_detect_vendor_from_tabular_unknown_raises(tmp_path: Path):
    from fasteit.parsers.detection import detect_vendor_from_tabular

    p = tmp_path / "unknown.txt"
    p.write_text("Some random content without known vendor keywords\n", encoding="latin1")
    with pytest.raises(ValueError, match="Could not detect vendor"):
        detect_vendor_from_tabular(p)
