"""Tests for Config dataclasses (Task 0.6.1)."""

import dataclasses

import pytest

from fasteit.config import (
    AnalysisConfig,
    Config,
    PreprocessingConfig,
    SegmentationConfig,
)


def test_config_instantiates_all_sections():
    cfg = Config()
    assert isinstance(cfg.preprocessing, PreprocessingConfig)
    assert isinstance(cfg.analysis, AnalysisConfig)
    assert isinstance(cfg.segmentation, SegmentationConfig)


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
        {"heart_rate_bpm": 0},
        {"lung_mask_threshold_fraction": 1.5},
        {"exclusion_margin_seconds": -1},
        {"ti_ratio_low": 0.7},
    ]:
        with pytest.raises(ValueError):
            PreprocessingConfig(**wrong)


def test_segmentation_config_defaults_follow_notebook_06():
    cfg = SegmentationConfig()
    assert (cfg.step_threshold, cfg.analysis_skip_seconds) == (1.0, 15.0)
    assert (cfg.analysis_min_breaths, cfg.analysis_max_breaths) == (5, 10)
    assert cfg.peep_levels is None


def test_segmentation_config_keeps_the_protocol_levels_as_a_tuple():
    cfg = SegmentationConfig(peep_levels=[16, 14, 12])
    assert cfg.peep_levels == (16.0, 14.0, 12.0)


def test_segmentation_config_rejects_impossible_values():
    assert SegmentationConfig(step_threshold=0.5).step_threshold == 0.5  # 1 cmH2O steps
    for wrong in [
        {"step_threshold": 0},
        {"analysis_min_breaths": 12},
        {"analysis_skip_seconds": -1},
        {"peep_levels": []},
        {"peep_levels": [16, float("nan")]},
    ]:
        with pytest.raises(ValueError):
            SegmentationConfig(**wrong)


def test_analysis_config_has_no_ri_levels_unless_given():
    cfg = AnalysisConfig()
    assert (cfg.ri_peep_high, cfg.ri_peep_low) == (None, None)
    assert (AnalysisConfig(ri_peep_high=14, ri_peep_low=4).ri_peep_high) == 14
    for wrong in [
        {"ri_peep_high": 14},
        {"ri_peep_low": 4},
        {"ri_peep_high": 4, "ri_peep_low": 14},
        {"ri_peep_high": 14, "ri_peep_low": 14},
    ]:
        with pytest.raises(ValueError):
            AnalysisConfig(**wrong)
