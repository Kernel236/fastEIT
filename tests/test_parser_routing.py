"""Template tests for auto-detection, parser routing, and batch loading."""

from pathlib import Path

import numpy as np
import pytest

from fasteit.parsers.detection import detect_vendor_and_format
from fasteit.parsers.draeger import DragerBinParser
from fasteit.parsers.draeger.bin.draeger_dtypes import FRAME_BASE_DTYPE
from fasteit.parsers.loader import build_parser_from_detection, load_folder, load_many


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
    p.write_text(
        "Some random content without known vendor keywords\n", encoding="latin1"
    )
    with pytest.raises(ValueError, match="Could not detect vendor"):
        detect_vendor_from_tabular(p)


# ── Vendor alias normalisation ────────────────────────────────────────────────


def test_vendor_alias_drager_typo_resolves(tmp_path: Path):
    """'drager' (missing a) must route to the Dräger bin parser."""
    from fasteit.parsers.detection import FileDetection

    det = FileDetection(path=tmp_path / "x.bin", extension=".bin", vendor="drager")
    parser = build_parser_from_detection(det)
    assert isinstance(parser, DragerBinParser)


def test_vendor_alias_umlaut_resolves(tmp_path: Path):
    """'dräger' (with umlaut) must route to the Dräger bin parser."""
    from fasteit.parsers.detection import FileDetection

    det = FileDetection(path=tmp_path / "x.bin", extension=".bin", vendor="dräger")
    parser = build_parser_from_detection(det)
    assert isinstance(parser, DragerBinParser)


# ── Batch loading ─────────────────────────────────────────────────────────────


def _write_bin(path: Path, n_frames: int = 5) -> Path:
    frames = np.zeros(n_frames, dtype=FRAME_BASE_DTYPE)
    dt = 1.0 / (50.0 * 86400.0)
    frames["ts"] = np.arange(n_frames) * dt
    frames.tofile(path)
    return path


def test_load_many_returns_list(tmp_path: Path):
    p1 = _write_bin(tmp_path / "a.bin")
    p2 = _write_bin(tmp_path / "b.bin")
    results = load_many([p1, p2])
    assert len(results) == 2
    assert all(r.n_frames == 5 for r in results)


def test_load_many_preserves_order(tmp_path: Path):
    p1 = _write_bin(tmp_path / "first.bin", n_frames=3)
    p2 = _write_bin(tmp_path / "second.bin", n_frames=7)
    results = load_many([p1, p2])
    assert results[0].n_frames == 3
    assert results[1].n_frames == 7


def test_load_folder_returns_matching_files(tmp_path: Path):
    _write_bin(tmp_path / "rec1.bin")
    _write_bin(tmp_path / "rec2.bin")
    (tmp_path / "notes.txt").write_text("ignore me")
    results = load_folder(tmp_path, pattern="*.bin")
    assert len(results) == 2


def test_load_folder_skips_unparseable_files(tmp_path: Path):
    _write_bin(tmp_path / "good.bin")
    (tmp_path / "bad.bin").write_bytes(b"\x00" * 100)  # unrecognised size
    results = load_folder(tmp_path, pattern="*.bin")
    assert len(results) == 1  # only the good file


def test_load_folder_empty_folder_returns_empty_list(tmp_path: Path):
    assert load_folder(tmp_path, pattern="*.bin") == []
