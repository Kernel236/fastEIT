"""Configuration dataclasses for the fasteit pipeline.

Single source of truth for tuneable parameters. Parsers and algorithms read
values from here - change once, propagates everywhere.

Structure:
    PreprocessingConfig - layer 2: cardiac filter, breaths, lung mask, regions
    AnalysisConfig      - layer 3: PEEP steps (to be filled)
    Config              - top-level container for pipeline sections

Rates and frequencies are given per minute here, as read on a monitor; the
preprocessing functions work in Hz and the pipeline converts.

Usage:
    from fasteit.config import Config

    cfg = Config()
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
            fast noise but leaves the cardiac component almost untouched (a 100/min
            oscillation keeps about 96 % of its amplitude at 50 Hz, 91 % at 20 Hz).
        respiratory_rate_min_bpm, respiratory_rate_max_bpm: Range (breaths/min) searched
            for the respiratory peak of the spectrum. Adult limits of eitprocessing
            ``RateDetection`` (v1.8.8; Somhorst et al., JOSS 2026;11(117):8179,
            DOI: 10.21105/joss.08179). Not valid for neonates, paediatric patients or
            high-frequency oscillation.
        spectrum_window_seconds: Length of the Welch window used for the respiratory
            rate (eitprocessing ``RateDetection`` default).
        heart_rate_bpm: Heart rate entered by the user (beats/min), e.g. from the
            monitor. If given, the MDN filter removes the cardiac component; if None,
            only the moving average is applied. The heart rate is never estimated
            from the data: the estimate was wrong in 2 of 11 recordings and a wrong
            heart rate close to the respiratory rate damages the breaths (section 5b).
            It is a snapshot: a heart rate that drifts during the recording, or an
            arrhythmia such as atrial fibrillation, is not removed by the MDN filter.
        mdn_notch_half_width_bpm, mdn_upper_limit_bpm, mdn_order: Multiple digital
            notch filter (Wisse et al., Physiol Meas 2024, DOI: 10.1088/1361-6579/ad46e3):
            half width of each band-stop around a cardiac harmonic, highest frequency
            filtered, and Butterworth order. Defaults of the eitprocessing ``MDNFilter``
            (v1.8.8).
        invalid_data_percentile, invalid_data_multiplier: A sample of the global signal
            is invalid when it is beyond mean ± multiplier × (distance between the mean
            and the given percentile, low or high). Defaults of eitprocessing
            ``BreathDetection`` (v1.8.8) (section 8).
        breath_baseline_seconds: Width of the Blackman moving average used as a local
            threshold: a peak must be above it and a valley below it, which lets the
            detection follow changes of EELI (eitprocessing default, section 8).
        breath_min_duration_seconds: Minimum time between two peaks and between two
            valleys (eitprocessing default 2/3 s; the original MATLAB scripts used
            about 0.67 s too).
        breath_low_amplitude_fraction: Peaks with an amplitude below this fraction of the
            median amplitude are merged into the breath around them (eitprocessing
            default).
        invalid_data_margin_seconds: A breath closer than this to invalid data gets
            ``flag_invalid_data`` (eitprocessing removes it; fastEIT keeps it, section 8).
    """

    moving_average_seconds: float = 0.1
    respiratory_rate_min_bpm: float = 6.0
    respiratory_rate_max_bpm: float = 60.0
    spectrum_window_seconds: float = 30.0
    heart_rate_bpm: float | None = None
    mdn_notch_half_width_bpm: float = 10.0
    mdn_upper_limit_bpm: float = 220.0
    mdn_order: int = 10
    invalid_data_percentile: float = 5.0
    invalid_data_multiplier: float = 4.0
    breath_baseline_seconds: float = 15.0
    breath_min_duration_seconds: float = 2 / 3
    breath_low_amplitude_fraction: float = 0.25
    invalid_data_margin_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.heart_rate_bpm is not None and self.heart_rate_bpm <= 0:
            raise ValueError(
                f"heart_rate_bpm must be positive, got {self.heart_rate_bpm}."
            )


@dataclass
class AnalysisConfig:
    """Parameters of layer 3 (PEEP step detection). To be filled.

    All defaults must cite supporting literature.
    """

    pass


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
