"""Tests for Config dataclasses (Task 0.6.1)."""

from fasteit.config import (
    DRAEGER,
    TIMPEL,
    AnalysisConfig,
    Config,
    DeviceConfig,
    PreprocessingConfig,
)


def test_device_config_defaults():
    cfg = DeviceConfig()
    assert cfg.name == "draeger"
    assert cfg.fs == 50.0
    assert cfg.pixel_grid == (32, 32)
    assert cfg.nan_value == -1e30
    assert cfg.frame_size_base == 4358
    assert cfg.frame_size_ext == 4382


def test_draeger_preset():
    assert DRAEGER.name == "draeger"
    assert DRAEGER.fs == 50.0
    assert DRAEGER.nan_value == -1e30
    assert DRAEGER.frame_size_base == 4358
    assert DRAEGER.frame_size_ext == 4382


def test_timpel_preset():
    assert TIMPEL.name == "timpel"
    assert TIMPEL.fs == 50.0
    assert TIMPEL.nan_value == -1000.0
    assert TIMPEL.frame_size_base is None
    assert TIMPEL.frame_size_ext is None


def test_config_instantiates_all_sections():
    cfg = Config()
    assert isinstance(cfg.device, DeviceConfig)
    assert isinstance(cfg.preprocessing, PreprocessingConfig)
    assert isinstance(cfg.analysis, AnalysisConfig)


def test_config_device_isolation():
    """Modifying one Config instance must not affect another."""
    cfg1 = Config()
    cfg2 = Config()
    cfg1.device.fs = 99.0
    assert cfg2.device.fs == 50.0
