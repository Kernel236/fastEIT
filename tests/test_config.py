"""Tests for Config dataclasses (Task 0.6.1)."""

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
