"""Configuration dataclasses for the fasteit pipeline.

Single source of truth for all tuneable parameters. Parsers and algorithms
read values from here — change once, propagates everywhere.

Structure:
    DeviceConfig        — hardware specs (stable, rarely changes)
    PreprocessingConfig — filter, lung mask, ROI params (TODO Task 0.6.1)
    AnalysisConfig      — breath detection, PEEP detection params (TODO Task 0.6.1)
    Config              — top-level container for all three sections

Usage:
    cfg = Config()                        # all defaults
    cfg = Config(device=DeviceConfig(fs=20.0))  # override one value

TODO (Task 0.6.2): Add evidence-based defaults with literature citations for
    PreprocessingConfig and AnalysisConfig once those phases are reached.
TODO (Task 0.6.3): Add preset_sensitive() and preset_permissive() factory
    functions for ARDS and spontaneous breathing use cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeviceConfig:
    """Hardware specifications for the Dräger PulmoVista 500.

    These values reflect the physical device and should not be changed
    unless targeting a different EIT system.
    """

    fs: float = 50.0
    """Acquisition frame rate in Hz. All file formats (.bin, .eit, .txt)
    are recorded at 50 Hz by the PulmoVista 500."""

    n_electrodes: int = 16
    """Number of electrodes on the belt."""

    n_measurements: int = 208
    """Independent transimpedance measurements per frame.
    16 injections × 13 measurements (adjacent-drive pattern,
    auto-measurements excluded).
    Ref: Frerichs I. et al., Thorax 2017;72:83-93
         DOI 10.1136/thoraxjnl-2016-208357"""

    pixel_grid: tuple[int, int] = (32, 32)
    """Reconstructed image size. PulmoVista outputs 32×32 pixel images."""

    frame_size_base: int = 4358
    """Byte size of a base frame in .bin files.
    Breakdown: ts1(4) + ts2(4) + dummy(4) + pixels(32×32×4=4096)
               + minmax_event(2×4=8) + event_text(30) + timing_error(4)
               + medibus_or_padding(208) = 4358 bytes."""

    frame_size_ext: int = 4382
    """Byte size of an extended frame in .bin files (with PressurePod).
    = frame_size_base (4358) + pressurepod_extra (24) = 4382 bytes.
    TODO (Task 1.3.4): verify pressurepod_extra field layout via xxd."""


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
    Each section is instantiated with its own defaults automatically.

    Example:
        cfg = Config()
        cfg.device.fs          # → 50.0
        cfg.device.n_electrodes  # → 16
    """

    device: DeviceConfig = field(default_factory=DeviceConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
