"""Tests for Drager ASC parser and tabular vendor detection."""

from pathlib import Path

import pytest

from fasteit.models.continuous_data import ContinuousSignalData
from fasteit.parsers.detection import detect_vendor_from_tabular
from fasteit.parsers.draeger import DragerAscParser
from fasteit.parsers.loader import load_data

# ── Synthetic fixture helpers ─────────────────────────────────────────────────


def _make_asc(tmp_path: Path, n_frames: int = 4) -> Path:
    """Write a minimal valid Dräger .asc file with >20 waveform columns."""
    col_names = [
        "Image",
        "Time",
        "Global",
        "MinMax",
        "Event",
        "EventText",
        "Timing_Error",
    ] + [f"Signal_{i}" for i in range(14)]
    header = "\t".join(col_names)  # 21 columns total

    # Time stored as fraction of day (same as .bin timestamps).
    # dt = 0.02 s = 0.02/86400 day → fs = 50 Hz after the parser converts back.
    _DT_DAYS = 0.02 / 86400.0
    rows = []
    for i in range(1, n_frames + 1):
        t = f"{i * _DT_DAYS:.10f}".replace(".", ",")
        vals = [str(i), t, f"{100 + i},50", "0", "0", "        ", "0"] + [
            f"{10 + i},0{i}" for _ in range(14)
        ]
        rows.append("\t".join(vals))

    meta = [
        "DraegerEIT Software V1.30",
        "File: synthetic.bin",
        f"Length: {n_frames} images = 0 s",
        "Dynamic image, time: 0,5",
        "LP/BP-Filter: LP",
        "Filter Cut-Off Frequ: 5 Hz",
        "",
    ]
    content = "\n".join(meta + [header] + rows) + "\n"
    p = tmp_path / "synthetic.asc"
    p.write_text(content, encoding="latin1")
    return p


# ── Synthetic tests (no patient file required) ────────────────────────────────


def test_asc_validate_returns_true_for_valid_file(tmp_path):
    p = _make_asc(tmp_path, n_frames=4)
    assert DragerAscParser().validate(p) is True


def test_asc_validate_returns_false_for_empty_file(tmp_path):
    p = tmp_path / "empty.asc"
    p.write_text("")
    assert DragerAscParser().validate(p) is False


def test_asc_validate_returns_false_for_wrong_extension(tmp_path):
    p = _make_asc(tmp_path, n_frames=4)
    p2 = p.rename(tmp_path / "recording.bin")
    assert DragerAscParser().validate(p2) is False


def test_asc_validate_returns_false_for_missing_file(tmp_path):
    assert DragerAscParser().validate(tmp_path / "ghost.asc") is False


def test_asc_parser_raises_on_empty_data_rows(tmp_path):
    """Valid >20-column header but no data rows → ValueError."""
    col_names = ["Image", "Time"] + [f"S_{i}" for i in range(19)]
    header = "\t".join(col_names)
    content = "File: x.bin\n\n" + header + "\n"
    p = tmp_path / "nodata.asc"
    p.write_text(content, encoding="latin1")
    with pytest.raises(ValueError, match="no data rows"):
        DragerAscParser().parse(p)


def test_asc_parser_synthetic_returns_continuous_signal_data(tmp_path):
    p = _make_asc(tmp_path, n_frames=4)
    data = DragerAscParser().parse(p)
    assert isinstance(data, ContinuousSignalData)
    assert data.file_format == "asc"
    assert data.n_frames == 4
    assert data.metadata["parsed_section"] == "continuous_waveforms"
    assert data.metadata["n_columns"] > 20


def test_asc_parser_synthetic_fs_estimated(tmp_path):
    p = _make_asc(tmp_path, n_frames=4)
    data = DragerAscParser().parse(p)
    # dt = 0.02 s → fs = 50 Hz
    assert data.fs is not None
    assert abs(data.fs - 50.0) < 1.0


def test_asc_parser_synthetic_header_metadata(tmp_path):
    p = _make_asc(tmp_path, n_frames=4)
    data = DragerAscParser().parse(p)
    assert data.metadata["source_eit_file"] == "synthetic.bin"
    assert data.metadata["declared_images"] == 4
    assert data.metadata["filter_mode"] == "LP"


def test_asc_parser_raises_if_no_waveform_header(tmp_path):
    # File with only 11-column header (no >20-column table)
    content = "File: x.bin\n\nImage\tTime\tA\tB\tC\tD\tE\tF\tG\tH\tI\n1\t0,02\t1\t2\t3\t4\t5\t6\t7\t8\t9\n"
    p = tmp_path / "bad.asc"
    p.write_text(content, encoding="latin1")
    with pytest.raises(ValueError, match="Could not find continuous waveform"):
        DragerAscParser().parse(p)


_ASC_FILE = Path("src/fasteit/test_files/patient01.asc")
_real_file_available = pytest.mark.skipif(
    not _ASC_FILE.exists(),
    reason="Real patient file not available (gitignored — run locally only)",
)


@_real_file_available
def test_detect_vendor_from_tabular_draeger_asc():
    vendor = detect_vendor_from_tabular(_ASC_FILE)
    assert vendor == "draeger"


@_real_file_available
def test_draeger_asc_parser_validate_true():
    parser = DragerAscParser()
    assert parser.validate(_ASC_FILE)


@_real_file_available
def test_draeger_asc_parser_parse_continuous_waveforms():
    parser = DragerAscParser()
    data = parser.parse(_ASC_FILE)

    assert isinstance(data, ContinuousSignalData)
    assert data.file_format == "asc"
    assert data.metadata.get("parsed_section") == "continuous_waveforms"
    assert data.n_frames > 1000
    assert "global" in data.table.columns
    assert "time" in data.table.columns
    assert "minmax" in data.table.columns or "min_max" in data.table.columns
    assert "timing_error" in data.table.columns
    assert data.metadata["n_columns"] > 20
    assert "declared_images" in data.metadata


@_real_file_available
def test_load_data_routes_asc_to_draeger_parser():
    data = load_data(_ASC_FILE)

    assert isinstance(data, ContinuousSignalData)
    assert data.vendor == "draeger"
    assert data.metadata.get("detected_extension") == ".asc"
