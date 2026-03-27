"""Tests for HeaderFormatSpec registry and get_eit_specs()."""

import pytest

from fasteit.parsers.header_formats import (
    DRAEGER_EIT_HEADER_SPEC,
    HEADER_FORMAT_SPECS,
    HeaderFormatSpec,
    get_eit_specs,
)


# ── Registry contents ─────────────────────────────────────────────────────────


def test_draeger_spec_in_registry():
    assert DRAEGER_EIT_HEADER_SPEC in HEADER_FORMAT_SPECS


def test_registry_is_nonempty():
    assert len(HEADER_FORMAT_SPECS) >= 1


def test_all_entries_are_header_format_spec():
    for spec in HEADER_FORMAT_SPECS:
        assert isinstance(spec, HeaderFormatSpec)


# ── DRAEGER_EIT_HEADER_SPEC field values ──────────────────────────────────────


def test_draeger_spec_vendor():
    assert DRAEGER_EIT_HEADER_SPEC.vendor == "draeger"


def test_draeger_spec_frame_size():
    assert DRAEGER_EIT_HEADER_SPEC.frame_size_bytes == 5495


def test_draeger_spec_n_electrodes():
    assert DRAEGER_EIT_HEADER_SPEC.n_electrodes == 16


def test_draeger_spec_n_measurements():
    assert DRAEGER_EIT_HEADER_SPEC.n_measurements == 208


def test_draeger_spec_magic_string_present():
    assert len(DRAEGER_EIT_HEADER_SPEC.magic_string) > 0


def test_draeger_spec_is_frozen():
    with pytest.raises((AttributeError, TypeError)):
        DRAEGER_EIT_HEADER_SPEC.vendor = "other"  # type: ignore[misc]


# ── get_eit_specs() ───────────────────────────────────────────────────────────


def test_get_eit_specs_draeger_returns_list():
    result = get_eit_specs("draeger")
    assert isinstance(result, list)
    assert len(result) >= 1


def test_get_eit_specs_draeger_contains_known_spec():
    result = get_eit_specs("draeger")
    assert DRAEGER_EIT_HEADER_SPEC in result


def test_get_eit_specs_unknown_vendor_raises():
    with pytest.raises(ValueError, match="No HeaderFormatSpec registered"):
        get_eit_specs("unknown_vendor_xyz")


def test_get_eit_specs_all_draeger_results_have_correct_vendor():
    for spec in get_eit_specs("draeger"):
        assert spec.vendor == "draeger"


def test_n_measurements_equals_16_times_13():
    """208 = 16 injections × 13 adjacent measurements (adjacent drive)."""
    spec = get_eit_specs("draeger")[0]
    assert spec.n_measurements == spec.n_electrodes * 13
