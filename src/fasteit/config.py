"""Configuration dataclasses for the fasteit pipeline.

Single source of truth for tuneable parameters. Parsers and algorithms read
values from here - change once, propagates everywhere.

Structure:
    PreprocessingConfig - layer 2: cardiac filter, breaths, lung mask, regions
    SegmentationConfig  - layer 2b: PEEP steps of a PEEP trial and the breaths analysed
    AnalysisConfig      - layer 3: clinical indices (to be filled)
    Config              - top-level container for pipeline sections

Rates and frequencies are given per minute here, as read on a monitor; the
preprocessing functions work in Hz and the pipeline converts.

Usage:
    from fasteit.config import Config

    cfg = Config()
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters of the preprocessing layer (layer 2).

    Frozen: a recording keeps the exact parameters it was made with. To change one,
    make a new config, e.g. ``dataclasses.replace(config, heart_rate_bpm=80)``.

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
            is invalid (a gross outlier: an artefact or an extreme manoeuvre) when it
            is beyond mean ± multiplier × (distance between the mean and the given
            percentile, low or high). Defaults of eitprocessing ``BreathDetection``
            (v1.8.8) (section 8).
        breath_baseline_seconds: Width of the Blackman moving average used as a local
            threshold: a peak must be above it and a valley below it. It follows slow
            EELI drift; breaths within about this time of a sudden EELI step (PEEP
            change) are unreliable. Below about 8 breaths/min the window starts to
            follow the breathing itself (eitprocessing default, section 8).
        breath_min_duration_seconds: Minimum time between two peaks and between two
            valleys, i.e. a maximum of 90 breaths/min (eitprocessing default 2/3 s; the
            original MATLAB scripts used about 0.67 s too).
        breath_low_amplitude_fraction: Peaks with an amplitude below this fraction of the
            median amplitude of the whole signal are merged into the breath around them
            (eitprocessing default). Merged peaks are not necessarily clinically
            irrelevant, e.g. hypopnoea in spontaneous breathing.
        invalid_data_margin_seconds: A breath closer than this to invalid data is
            flagged (``near_invalid`` of ``detect_breaths``); eitprocessing removes it,
            fastEIT keeps it (section 8).
        breath_short_seconds, breath_long_seconds: A breath shorter or longer than this
            is flagged (``flag_short``, ``flag_long``). Thresholds of the fastEIT
            roadmap (task 4.11). 0.5 s is 120 breaths/min, far above the adult range
            and below the 2/3 s minimum of the detection, so it mainly catches
            detection artefacts. A breath above 20 s is a very long cycle: a pause or
            apnoea, or a missed breath (section 8).
        ti_ratio_low, ti_ratio_high: A breath with Ti/Ttot outside this range is
            flagged (``flag_suspect_ti_ratio``). Ti/Ttot here is the rise fraction of
            the impedance curve (valley to peak over valley to valley): it only
            approximates the clinical Ti/Ttot, so the range is a plausibility check on
            the detection, not a clinical judgement. Wide limits around the usual
            0.3-0.5; obstructive or fast-breathing patients can fall outside without an
            error (roadmap task 4.11, section 8).
        local_window_breaths: Number of breaths, centred on each breath and including
            it, whose median is the reference for the outlier flags. 21 breaths are
            about 40 s at 30 breaths/min (about 2 min at 10 breaths/min): long enough
            to ignore one event, short enough to follow a PEEP change (section 8).
        local_min_breaths: Fewest breaths in the window to compute the local median
            (at the start and end of a short recording); with fewer, no outlier flag.
        amplitude_ratio_low, amplitude_ratio_high: A breath with an amplitude below
            or above these times the local median is flagged
            (``flag_amplitude_outlier``). Ratios were preferred to a robust z-score,
            which flagged normal variability in regular recordings (section 8).
        duration_ratio_high: A breath longer than this times the local median duration
            is flagged (``flag_duration_outlier``), e.g. two breaths merged into one.
        lung_mask_threshold_fraction: A pixel is lung when its mean TIV (and, for the
            watershed, its amplitude) is at least this fraction of the maximum TIV.
            Default of eitprocessing ``WatershedLungspace`` (v1.8.8; Somhorst et al.,
            JOSS 2026, DOI: 10.21105/joss.08179) (sections 7, 7b).
        lung_mask_min_region_pixels: Islands of the lung mask smaller than this are
            removed (eitprocessing ``FilterROIBySize`` default, section 9).
        heart_rate_warning_low_bpm, heart_rate_warning_high_bpm: A heart rate entered
            outside this range gives a warning in the log (adult range of
            eitprocessing ``RateDetection``).
        heart_rate_warning_margin_bpm: A heart rate closer than this to 1, 2 or 3
            times the respiratory rate gives a warning: the MDN notches then fall on
            the breathing (section 5b). Half width of the notch (10/min) plus a
            heuristic 2/min, so that the notch edge stays clear of the harmonic.
        exclusion_margin_seconds: Every exclusion (manual or from the device) is
            widened by this margin on both sides, because the forward-backward
            filters spread an artifact over the neighbouring samples. Same width as
            the margin around invalid data in eitprocessing ``BreathDetection``. The
            narrow MDN notches can ring longer, at the heart rate and with a small
            amplitude; the first and last seconds of every recording also carry
            filter edge effects and are not excluded.
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
    breath_short_seconds: float = 0.5
    breath_long_seconds: float = 20.0
    ti_ratio_low: float = 0.2
    ti_ratio_high: float = 0.6
    local_window_breaths: int = 21
    local_min_breaths: int = 5
    amplitude_ratio_low: float = 0.5
    amplitude_ratio_high: float = 2.0
    duration_ratio_high: float = 2.0
    lung_mask_threshold_fraction: float = 0.15
    lung_mask_min_region_pixels: int = 10
    heart_rate_warning_low_bpm: float = 40.0
    heart_rate_warning_high_bpm: float = 200.0
    heart_rate_warning_margin_bpm: float = 12.0
    exclusion_margin_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.heart_rate_bpm is not None and self.heart_rate_bpm <= 0:
            raise ValueError(
                f"heart_rate_bpm must be positive, got {self.heart_rate_bpm}."
            )
        if not 0 < self.lung_mask_threshold_fraction < 1:
            raise ValueError("lung_mask_threshold_fraction must be between 0 and 1.")
        if self.lung_mask_min_region_pixels < 1:
            raise ValueError("lung_mask_min_region_pixels must be at least 1.")
        if self.exclusion_margin_seconds < 0:
            raise ValueError("exclusion_margin_seconds must not be negative.")
        if not self.ti_ratio_low < self.ti_ratio_high:
            raise ValueError("ti_ratio_low must be below ti_ratio_high.")


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters of the PEEP trial segmentation (layer 2b).

    Evidence and decisions: ``notebooks/06_peep_trial_segmentation.ipynb`` (section in
    brackets). These are fastEIT choices tested on four decremental trials, not values
    from a paper. Pressures are in the units of the recording (mbar for Draeger, cmH2O
    for Timpel; 1 mbar = 1.02 cmH2O).

    Attributes:
        pressure_channels: Airway pressure waveforms searched in ``EITRecording.aux``,
            in order of preference: Draeger PressurePod, then the Medibus or Timpel
            ``airway_pressure`` (section 2).
        peep_value_channel, plateau_value_channel: Values computed by the ventilator
            (Medibus), used only if there is no pressure waveform: the PEEP and plateau
            of a breath are their median over the breath (section 7).
        peep_seconds: The PEEP of a breath is the median airway pressure in this time
            before the start of inspiration found on the pressure curve (section 2).
        plateau_window_seconds: The plateau is the median airway pressure from the first
            to the second value before the start of expiration; it needs an
            end-inspiratory pause (section 4).
        step_threshold: A new PEEP step starts when ``step_min_breaths`` breaths in a
            row are further than this from the current level: half of the step of the
            protocol, 1 for steps of 2 (section 3).
        step_min_breaths: Breaths in a row needed to open a step, so that a single odd
            breath (a maneuver) cannot (section 3).
        step_agreement: Those breaths must also be within this of each other (section 3).
        analysis_skip_seconds: In each step the breaths in this time after the change of
            PEEP are not analysed, as EELI is still moving (section 5).
        analysis_max_breaths, analysis_min_breaths: Then the last valid breaths of the
            step are analysed, up to ``analysis_max_breaths``; fewer than
            ``analysis_min_breaths`` give a warning (section 5).
    """

    pressure_channels: tuple[str, ...] = ("airway_pressure_pod", "airway_pressure")
    peep_value_channel: str = "peep"
    plateau_value_channel: str = "plateau_pressure"
    peep_seconds: float = 0.2
    plateau_window_seconds: tuple[float, float] = (0.25, 0.05)
    step_threshold: float = 1.0
    step_min_breaths: int = 3
    step_agreement: float = 1.0
    analysis_skip_seconds: float = 15.0
    analysis_max_breaths: int = 10
    analysis_min_breaths: int = 5

    def __post_init__(self) -> None:
        if not self.step_threshold > 0:
            raise ValueError("step_threshold must be positive.")
        if self.step_min_breaths < 1:
            raise ValueError("step_min_breaths must be at least 1.")
        if not 1 <= self.analysis_min_breaths <= self.analysis_max_breaths:
            raise ValueError(
                "analysis_min_breaths must be between 1 and analysis_max_breaths."
            )
        first, second = self.plateau_window_seconds
        if not first > second >= 0:
            raise ValueError(
                "plateau_window_seconds must be (start, end) with start > end >= 0."
            )


@dataclass
class AnalysisConfig:
    """Parameters of layer 3 (clinical indices). To be filled.

    All defaults must cite supporting literature.
    """

    pass


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
