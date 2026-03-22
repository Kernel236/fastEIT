"""Tests for Drager ASC parser and tabular vendor detection."""

from pathlib import Path

from fasteit.models.continuous_data import ContinuousSignalData
from fasteit.parsers.detection import detect_vendor_from_tabular
from fasteit.parsers.draeger import DragerAscParser
from fasteit.parsers.loader import load_data


def _asc_file() -> Path:
    return Path("src/fasteit/test_files/patient01.asc")


def test_detect_vendor_from_tabular_draeger_asc():
    vendor = detect_vendor_from_tabular(_asc_file())
    assert vendor == "draeger"


def test_draeger_asc_parser_validate_true():
    parser = DragerAscParser()
    assert parser.validate(_asc_file())


def test_draeger_asc_parser_parse_continuous_waveforms():
    parser = DragerAscParser()
    data = parser.parse(_asc_file())

    assert isinstance(data, ContinuousSignalData)
    assert data.file_format == "asc"
    assert data.metadata.get("parsed_section") == "continuous_waveforms"
    # File has 11500 frames; all should be loaded
    assert data.n_frames > 1000
    # Core EIT columns
    assert "global" in data.table.columns
    assert "time" in data.table.columns
    # Event and timing columns present in second table (not in Tidal Variations)
    assert "minmax" in data.table.columns or "min_max" in data.table.columns
    assert "timing_error" in data.table.columns
    # Schema has >20 columns even if many are NaN (Medibus not always connected)
    assert data.metadata["n_columns"] > 20
    # Header metadata preserved
    assert "declared_images" in data.metadata


def test_load_data_routes_asc_to_draeger_parser():
    data = load_data(_asc_file())

    assert isinstance(data, ContinuousSignalData)
    assert data.vendor == "draeger"
    assert data.metadata.get("detected_extension") == ".asc"
