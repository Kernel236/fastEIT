"""Configuration dataclasses for the fasteit pipeline.

Single source of truth for tuneable parameters. Parsers and algorithms read
values from here - change once, propagates everywhere.

Structure:
    PreprocessingConfig - filter, lung mask, ROI params (TODO Task 0.6.1)
    AnalysisConfig      - breath detection, PEEP detection params (TODO Task 0.6.1)
    Config              - top-level container for pipeline sections

Usage:
    from fasteit.config import Config

    cfg = Config()
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pipeline configs (populated incrementally in Fase 4-5)
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Parameters of the preprocessing layer (layer 2).

    Every default states where it comes from. The evidence for each choice is in
    ``notebooks/05_preprocessing_choices.ipynb`` (section given in brackets).

    Attributes:
        moving_average_seconds: Width of the centred moving average used when no heart
            rate is given, so no cardiac filter is applied (section 5c). It is the
            ``smooth()`` of the original MATLAB analysis scripts, defined in seconds
            instead of samples so that it behaves the same at 20 and 50 Hz. It removes
            fast noise but leaves the cardiac component almost untouched.
    """

    moving_average_seconds: float = 0.1


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
