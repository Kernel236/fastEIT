"""Tests for BaseParser ABC (Task 0.5.1)."""

from pathlib import Path

import pytest

from fasteit.models.base_data import BaseData
from fasteit.parsers.base import BaseParser

# ── Concrete stub for testing ─────────────────────────────────────────────


class _AlwaysValidParser(BaseParser):
    """Minimal concrete parser that always passes validation."""

    def validate(self, path: Path) -> bool:
        return True

    def parse(self, path: Path) -> BaseData:
        return BaseData(filename=str(path), file_format="bin")


class _AlwaysInvalidParser(BaseParser):
    """Minimal concrete parser that always fails validation."""

    def validate(self, path: Path) -> bool:
        return False

    def parse(self, path: Path) -> BaseData:  # pragma: no cover
        return BaseData()


# ── Tests ─────────────────────────────────────────────────────────────────


def test_base_parser_cannot_be_instantiated():
    """BaseParser is abstract — direct instantiation must raise TypeError."""
    with pytest.raises(TypeError):
        BaseParser()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    parser = _AlwaysValidParser()
    assert isinstance(parser, BaseParser)


def test_parse_safe_raises_on_missing_file(tmp_path):
    parser = _AlwaysValidParser()
    missing = tmp_path / "nonexistent.bin"
    with pytest.raises(FileNotFoundError, match="File not found"):
        parser.parse_safe(missing)


def test_parse_safe_raises_on_invalid_format(tmp_path):
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\x00" * 10)
    parser = _AlwaysInvalidParser()
    with pytest.raises(ValueError, match="invalid or unsupported"):
        parser.parse_safe(f)


def test_parse_safe_returns_data_on_valid_file(tmp_path):
    f = tmp_path / "ok.bin"
    f.write_bytes(b"\x00" * 10)
    parser = _AlwaysValidParser()
    result = parser.parse_safe(f)
    assert isinstance(result, BaseData)
    assert result.filename == str(f)


def test_parse_safe_accepts_string_path(tmp_path):
    """parse_safe should coerce str → Path internally."""
    f = tmp_path / "ok.bin"
    f.write_bytes(b"\x00" * 10)
    parser = _AlwaysValidParser()
    result = parser.parse_safe(str(f))  # str, not Path
    assert result.filename == str(f)
