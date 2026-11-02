"""Configuration dataclasses for the fasteit pipeline.

Single source of truth for tuneable parameters. Parsers and algorithms read
values from here - change once, propagates everywhere.

Structure:
    PreprocessingConfig - layer 2: cardiac filter, breaths, lung mask, regions, pressures
                          and tidal volume of each breath
    SegmentationConfig  - layer 2b: PEEP steps of a PEEP trial and the breaths analysed
    AnalysisConfig      - layer 3: clinical indices (recruitment-to-inflation ratio)
    Config              - top-level container for pipeline sections

Rates and frequencies are given per minute here, as read on a monitor; the
preprocessing functions work in Hz and the pipeline converts.

Usage:
    from fasteit.config import Config

    cfg = Config()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters of the preprocessing layer (layer 2).

    Frozen: a recording keeps the exact parameters it was made with. To change one,
    make a new config, e.g. ``dataclasses.replace(config, heart_rate_bpm=80)``.

    Only the choices a study may change are here; the details of the methods taken from
    eitprocessing or fixed by the evidence are constants of their modules
    (``cardiac_filter``, ``breaths``, ``breath_quality``, ``lung_mask``,
    ``breath_pressures``, ``tidal_volume``). Evidence:
    ``notebooks/05_preprocessing_choices.ipynb`` (section in brackets).

    Attributes:
        heart_rate_bpm: Heart rate entered by the user (beats/min), e.g. from the
            monitor. If given, the MDN filter removes the cardiac component; if None,
            only the moving average is applied. The heart rate is never estimated
            from the data: the estimate was wrong in 2 of 11 recordings and a wrong
            heart rate close to the respiratory rate damages the breaths (section 5b).
            It is a snapshot: a heart rate that drifts during the recording, or an
            arrhythmia such as atrial fibrillation, is not removed by the MDN filter.
        moving_average_seconds: Width of the centred moving average used when no heart
            rate is given (section 5c): the ``smooth()`` of the original MATLAB analysis
            scripts, in seconds so that it behaves the same at 20 and 50 Hz. It leaves
            the cardiac component almost untouched (a 100/min oscillation keeps about
            96 % of its amplitude at 50 Hz, 91 % at 20 Hz).
        breath_short_seconds, breath_long_seconds: A breath shorter or longer than this
            is flagged (``flag_short``, ``flag_long``; roadmap task 4.11). 0.5 s is
            120 breaths/min, below the 2/3 s minimum of the detection, so it mainly
            catches detection artefacts; above 20 s: a pause, an apnoea or a missed
            breath (section 8).
        ti_ratio_low, ti_ratio_high: A breath with Ti/Ttot outside this range is
            flagged (``flag_suspect_ti_ratio``). Ti/Ttot here is the rise fraction of
            the impedance curve, a plausibility check on the detection, not a clinical
            judgement (roadmap task 4.11, section 8).
        local_window_breaths: Breaths, centred on each breath, whose median is the
            reference for the outlier flags: 21 are about 40 s at 30 breaths/min, long
            enough to ignore one event, short enough to follow a PEEP change (section 8).
        amplitude_ratio_low, amplitude_ratio_high: A breath with an amplitude below
            or above these times the local median is flagged
            (``flag_amplitude_outlier``; section 8).
        duration_ratio_high: A breath longer than this times the local median duration
            is flagged (``flag_duration_outlier``), e.g. two breaths merged into one.
        lung_mask_threshold_fraction: A pixel is lung when its mean TIV (and, for the
            watershed, its amplitude) is at least this fraction of the maximum TIV
            (eitprocessing ``WatershedLungspace`` default; sections 7, 7b).
        exclusion_margin_seconds: Every exclusion (manual or from the device) is
            widened by this margin on both sides, because the forward-backward filters
            spread an artifact over the neighbouring samples (the margin around invalid
            data of eitprocessing ``BreathDetection``).
    """

    heart_rate_bpm: float | None = None
    moving_average_seconds: float = 0.1
    breath_short_seconds: float = 0.5
    breath_long_seconds: float = 20.0
    ti_ratio_low: float = 0.2
    ti_ratio_high: float = 0.6
    local_window_breaths: int = 21
    amplitude_ratio_low: float = 0.5
    amplitude_ratio_high: float = 2.0
    duration_ratio_high: float = 2.0
    lung_mask_threshold_fraction: float = 0.15
    exclusion_margin_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.heart_rate_bpm is not None and self.heart_rate_bpm <= 0:
            raise ValueError(
                f"heart_rate_bpm must be positive, got {self.heart_rate_bpm}."
            )
        if not 0 < self.lung_mask_threshold_fraction < 1:
            raise ValueError("lung_mask_threshold_fraction must be between 0 and 1.")
        if self.exclusion_margin_seconds < 0:
            raise ValueError("exclusion_margin_seconds must not be negative.")
        if not self.ti_ratio_low < self.ti_ratio_high:
            raise ValueError("ti_ratio_low must be below ti_ratio_high.")


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters of the PEEP trial segmentation (layer 2b).

    Protocol choices tested on four decremental trials
    (``notebooks/06_peep_trial_segmentation.ipynb``, section in brackets); pressures in
    cmH2O. The details of the step finding are constants of ``peep_steps``.

    Attributes:
        step_threshold: Half of the PEEP step of the protocol (1 for steps of 2): a new
            step starts when 3 breaths in a row are further than this from the current
            level and within this of each other; a breath is analysed only if its PEEP
            is within half of this of the level of its step (section 3, 5).
        analysis_skip_seconds: In each step the breaths in this time after the change of
            PEEP are not analysed, as EELI is still moving (section 5).
        analysis_max_breaths, analysis_min_breaths: Then the last valid breaths of the
            step are analysed, up to ``analysis_max_breaths``; fewer than
            ``analysis_min_breaths`` give a warning (section 5).
        peep_levels: PEEP levels of the protocol, e.g. ``(16, 14, ..., 0)``. Each step
            gets the nearest one as ``peep_set``, after removing the offset common to
            all steps, so that steps can be compared between patients (the measured
            PEEP is noisy, and 10 mbar set on a Dräger ventilator is 10.2 cmH2O);
            none, with a warning, if further than ``step_threshold``. None:
            ``peep_set`` is empty.
    """

    step_threshold: float = 1.0
    analysis_skip_seconds: float = 15.0
    analysis_max_breaths: int = 10
    analysis_min_breaths: int = 5
    peep_levels: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.step_threshold > 0:
            raise ValueError("step_threshold must be positive.")
        if not 1 <= self.analysis_min_breaths <= self.analysis_max_breaths:
            raise ValueError(
                "analysis_min_breaths must be between 1 and analysis_max_breaths."
            )
        if self.analysis_skip_seconds < 0:
            raise ValueError("analysis_skip_seconds must not be negative.")
        if self.peep_levels is not None:
            levels = tuple(float(level) for level in self.peep_levels)
            if not levels or not all(math.isfinite(level) for level in levels):
                raise ValueError("peep_levels must be finite numbers.")
            # A list given by the user is stored as a tuple: the config stays immutable.
            object.__setattr__(self, "peep_levels", levels)


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters of layer 3 (clinical indices).

    The definitions of the indices are fixed in ``notebooks/07_clinical_indices.ipynb``;
    only the recruitment-to-inflation ratio has parameters.

    Attributes:
        ri_peep_high, ri_peep_low, ri_min_peep_drop: The R/I of a PEEP trial compares the
            step nearest to ``ri_peep_high`` with the step nearest to ``ri_peep_low``
            among those at least ``ri_min_peep_drop`` below it (cmH2O). 15 and 5 are the
            levels of the original method (Chen et al., Am J Respir Crit Care Med
            2020;201:178-187, DOI: 10.1164/rccm.201902-0334OC).
    """

    ri_peep_high: float = 15.0
    ri_peep_low: float = 5.0
    ri_min_peep_drop: float = 10.0

    def __post_init__(self) -> None:
        if not self.ri_min_peep_drop > 0:
            raise ValueError("ri_min_peep_drop must be positive.")
        if not self.ri_peep_high > self.ri_peep_low:
            raise ValueError("ri_peep_high must be above ri_peep_low.")


@dataclass
class Config:
    """Top-level configuration container for the fasteit pipeline.

    Aggregates PreprocessingConfig, SegmentationConfig and AnalysisConfig.

    Example:
        cfg = Config()
        cfg.preprocessing
        cfg.segmentation
        cfg.analysis
    """

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
