"""Tests for Config dataclasses (Task 0.6.1)."""

import dataclasses

import pytest

from fasteit.config import (
    AnalysisConfig,
    Config,
    PreprocessingConfig,
)


def test_config_instantiates_all_sections():
    cfg = Config()
    assert isinstance(cfg.preprocessing, PreprocessingConfig)
    assert isinstance(cfg.analysis, AnalysisConfig)


def test_config_section_isolation():
    """Each Config instance must own independent section instances."""
    cfg1 = Config()
    cfg2 = Config()

    assert cfg1.preprocessing is not cfg2.preprocessing
    assert cfg1.analysis is not cfg2.analysis


def test_preprocessing_config_is_frozen():
    cfg = PreprocessingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.heart_rate_bpm = 80
    assert dataclasses.replace(cfg, heart_rate_bpm=80).heart_rate_bpm == 80


def test_preprocessing_config_rejects_impossible_values():
    for wrong in [
        {"lung_mask_threshold_fraction": 1.5},
        {"lung_mask_min_region_pixels": 0},
        {"exclusion_margin_seconds": -1},
        {"ti_ratio_low": 0.7},
    ]:
        with pytest.raises(ValueError):
            PreprocessingConfig(**wrong)
