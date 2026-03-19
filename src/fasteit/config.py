"""Configuration dataclasses for the fasteit pipeline.

Single source of truth for all tuneable parameters. Parsers and algorithms
read values from here — change once, propagates everywhere.

Structure:
    DeviceConfig        — hardware specs (stable, rarely changes)
    DRAEGER             — preset for Dräger PulmoVista 500
    TIMPEL              — preset for Timpel Enlight 2100
    PreprocessingConfig — filter, lung mask, ROI params (TODO Task 0.6.1)
    AnalysisConfig      — breath detection, PEEP detection params (TODO Task 0.6.1)
    Config              — top-level container for all three sections

Usage:
    from fasteit.config import DRAEGER, Config

    cfg = Config()                             # defaults to Dräger
    cfg = Config(device=DRAEGER)               # explicit preset
    cfg = Config(device=DeviceConfig(name="draeger", fs=20.0))  # override
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeviceConfig:
    """Hardware specifications for one EIT device.

    Device-agnostic container. Use the module-level presets (DRAEGER, TIMPEL)
    rather than constructing this directly.

    Attributes:
        name:            Device identifier string ("draeger", "timpel", ...).
        fs:              Acquisition frame rate in Hz.
        pixel_grid:      Reconstructed image size (rows, cols).
        nan_value:       Sentinel for missing/invalid data in raw files.
                         Dräger uses -1e30; Timpel uses -1000.0.
        frame_size_base: Byte size of one base frame in binary formats.
                         None for text/CSV devices (e.g. Timpel).
        frame_size_ext:  Byte size of an extended frame (with PressurePod).
                         None for devices without a binary extension mode.
    """

    name: str = "draeger"
    fs: float = 50.0
    pixel_grid: tuple[int, int] = (32, 32)
    nan_value: float = -1e30
    frame_size_base: int | None = 4358
    frame_size_ext: int | None = 4382


# ---------------------------------------------------------------------------
# Device presets
# ---------------------------------------------------------------------------

DRAEGER = DeviceConfig(
    name="draeger",
    fs=50.0,
    pixel_grid=(32, 32),
    nan_value=-1e30,
    frame_size_base=4358,
    frame_size_ext=4382,
)
"""Preset for Dräger PulmoVista 500.

Binary .bin format. Frame sizes confirmed via numpy structured dtype analysis.
TODO (Task 1.1.1): verify frame_size_ext pressurepod layout via xxd.
"""

TIMPEL = DeviceConfig(
    name="timpel",
    fs=50.0,
    pixel_grid=(32, 32),
    nan_value=-1000.0,
    frame_size_base=None,
    frame_size_ext=None,
)
"""Preset for Timpel Enlight 2100.

CSV format — no binary frame size. 1030 columns per frame:
  cols 0–1023: pixel impedance, cols 1024–1029: airway pressure, flow,
  volume, min-flag, max-flag, QRS marker.
TODO (Task 1.x): implement TimpelParser once real files are available.
"""


# ---------------------------------------------------------------------------
# Pipeline configs (populated incrementally in Fase 4–5)
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Parameters for signal filtering, lung mask, and ROI extraction.

    TODO (Task 0.6.1 / Fase 4): populate fields when implementing:
        - Butterworth filter (Task 4.1.1)
        - Lung mask (Task 4.4.1)
        - ROI definitions (Task 4.8.1)
    All defaults must cite supporting literature.
    """

    pass  # fields added in Fase 4


@dataclass
class AnalysisConfig:
    """Parameters for breath detection and PEEP step detection.

    TODO (Task 0.6.1 / Fase 4-5): populate fields when implementing:
        - Breath detection (Task 4.10.1)
        - PEEP detection (Task 5.7.1)
    All defaults must cite supporting literature.
    """

    pass  # fields added in Fase 5


@dataclass
class Config:
    """Top-level configuration container for the fasteit pipeline.

    Aggregates DeviceConfig, PreprocessingConfig, and AnalysisConfig.

    Example:
        cfg = Config()
        cfg.device.fs          # → 50.0
        cfg.device.nan_value   # → -1e30
    """

    device: DeviceConfig = field(default_factory=DeviceConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
