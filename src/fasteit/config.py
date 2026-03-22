"""Configuration dataclasses for the fasteit pipeline.

Single source of truth for tuneable parameters. Parsers and algorithms read
values from here — change once, propagates everywhere.

Structure:
    PreprocessingConfig — filter, lung mask, ROI params (TODO Task 0.6.1)
    AnalysisConfig      — breath detection, PEEP detection params (TODO Task 0.6.1)
    Config              — top-level container for pipeline sections

Usage:
    from fasteit.config import Config

    cfg = Config()
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

    Aggregates PreprocessingConfig and AnalysisConfig.

    Example:
        cfg = Config()
        cfg.preprocessing
        cfg.analysis
    """

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
