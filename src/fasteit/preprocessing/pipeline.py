"""preprocess(): from the parsed frames to a recording ready for analysis (layer 2).

Fixed order of the steps (notebooks/05_preprocessing_choices.ipynb):
1. time in seconds, pixel copy with the device "no data" values as NaN;
2. frames with missing pixel data filled and excluded;
3. respiratory rate from the spectrum, cardiac step (MDN with a heart rate given by
   the user, otherwise only a moving average);
4. breath detection on the global signal (sum of all pixels);
5. quality flags and exclusions of each breath;
6. lung mask from the valid breaths, regions of interest, regional signals;
7. auxiliary signals (pressures, flow, device markers) with "no data" as NaN.

Every step writes what it did in ``EITRecording.log`` (saved with the results) and
in the Python logger ``fasteit.preprocessing.pipeline`` (technical log, configured by
the application, not here).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fasteit import __version__
from fasteit.config import PreprocessingConfig
from fasteit.models.eit_recording import EITRecording, RecordingContext, Segment
from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.draeger.bin.bin_utils import replace_draeger_sentinels
from fasteit.preprocessing.auxiliary import auxiliary_signals
from fasteit.preprocessing.breath_quality import quality_flags
from fasteit.preprocessing.breaths import detect_breaths
from fasteit.preprocessing.cardiac_filter import (
    SECONDS_PER_MINUTE,
    heart_rate_warnings,
    mdn_filter,
    moving_average,
    respiratory_rate,
)
from fasteit.preprocessing.exclusions import (
    breaths_in_segments,
    fill_missing_frames,
    frames_to_segments,
    merge_overlapping,
    missing_frames,
    widen,
)
from fasteit.preprocessing.lung_mask import remove_small_regions, watershed_mask
from fasteit.preprocessing.regions import lung_regions
from fasteit.run_log import RunLog

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0
NOT_REMOVED = (
    "cardiac component NOT removed (moving average of {seconds} s only); "
    "pixel-level results and small tidal volumes may be affected."
)


def recording_time(raw: ReconstructedFrameData) -> np.ndarray:
    """Time of each frame in seconds from the start of the recording.

    Draeger ``.bin`` timestamps are a fraction of the day: they are converted to
    seconds and a recording that crosses midnight is unwrapped. Other vendors already
    give seconds.
    """
    timestamps = np.asarray(raw.timestamps, dtype=float)
    if raw.vendor == "draeger":
        seconds = timestamps * SECONDS_PER_DAY
        # Each jump back of more than half a day is a midnight crossing.
        days_passed = np.cumsum(np.r_[0, np.diff(seconds) < -SECONDS_PER_DAY / 2])
        timestamps = seconds + days_passed * SECONDS_PER_DAY
    return timestamps - timestamps[0]


def pixel_values(raw: ReconstructedFrameData) -> np.ndarray:
    """Float copy of the pixels, with the Draeger "no data" values as NaN.

    The parsed pixels are never modified (they can be a read-only memory map).
    Timpel pixels already have NaN from the parser.
    """
    if raw.vendor == "draeger":
        return replace_draeger_sentinels(raw.pixels).astype(float)
    return np.array(raw.pixels, dtype=float)


def _check_input(raw: ReconstructedFrameData) -> None:
    """Raise ValueError if ``raw`` cannot be preprocessed."""
    if not isinstance(raw, ReconstructedFrameData) or raw.frames is None:
        raise ValueError("preprocess() needs ReconstructedFrameData with frames.")
    if raw.fs is None or not raw.fs > 0:
        raise ValueError(f"Sampling frequency must be positive, got {raw.fs}.")
    if raw.pixels.ndim != 3:
        raise ValueError(f"Pixels must be (N, rows, cols), got {raw.pixels.shape}.")


def _device_exclusions(
    raw: ReconstructedFrameData, time: np.ndarray, missing: np.ndarray, log: RunLog
) -> list[Segment]:
    """Exclusions from the device: missing pixel data, timing errors, time gaps.

    Raises:
        ValueError: if the time does not always increase.
    """
    fs = raw.fs
    steps = np.diff(time)
    if np.any(steps <= 0):
        raise ValueError("Frame times do not always increase: check the file.")

    segments = frames_to_segments(time, missing, fs, "missing_data")
    if missing.any():
        log.warning(
            f"{missing.sum()} frames with missing pixel data filled and excluded."
        )

    if "timing_error" in raw.frames.dtype.names:
        timing_error = raw.frames["timing_error"] != 0
        segments += frames_to_segments(time, timing_error, fs, "timing_error")
        if timing_error.any():
            log.warning(
                f"{timing_error.sum()} frames with a device timing error excluded."
            )

    # Dropped frames: the filters treat the samples as evenly spaced, so the breath
    # across a gap is excluded.
    gaps = np.flatnonzero(steps > 1.5 / fs)
    segments += [
        Segment(float(time[i]), float(time[i + 1]), label="time_gap", source="device")
        for i in gaps
    ]
    if len(gaps):
        log.warning(f"{len(gaps)} gaps in the frame times (dropped frames) excluded.")
    return segments


def _respiratory_rate(
    pixels: np.ndarray, fs: float, config: PreprocessingConfig, log: RunLog
) -> float | None:
    """Respiratory rate (Hz) from the spectrum of all pixels, or None."""
    try:
        rr_hz = respiratory_rate(
            np.nansum(pixels, axis=(1, 2)),
            fs,
            min_bpm=config.respiratory_rate_min_bpm,
            max_bpm=config.respiratory_rate_max_bpm,
            window_seconds=config.spectrum_window_seconds,
        )
    except ValueError as error:
        log.warning(f"Respiratory rate not estimated: {error}")
        return None
    rr_bpm = rr_hz * SECONDS_PER_MINUTE
    log.info(f"Respiratory rate from the spectrum: {rr_bpm:.1f}/min.")
    if rr_bpm in (config.respiratory_rate_min_bpm, config.respiratory_rate_max_bpm):
        log.warning(
            "Respiratory rate on the edge of the searched range: probably a slow drift "
            "of the signal, not the breathing."
        )
    return rr_hz


def _cardiac_step(
    pixels: np.ndarray,
    fs: float,
    rr_hz: float | None,
    config: PreprocessingConfig,
    log: RunLog,
) -> tuple[np.ndarray, str]:
    """Cardiac filter policy of section 5c.

    Returns:
        The pixels and the filter applied: ``"mdn"`` or ``"moving_average"``.
    """
    hr_bpm = config.heart_rate_bpm
    not_removed = NOT_REMOVED.format(seconds=config.moving_average_seconds)
    if hr_bpm is None:
        log.warning(f"No heart rate given: {not_removed}")
    elif rr_hz is None:
        log.warning(f"MDN filter needs the respiratory rate: {not_removed}")
    else:
        for message in heart_rate_warnings(
            hr_bpm,
            rr_hz * SECONDS_PER_MINUTE,
            low_bpm=config.heart_rate_warning_low_bpm,
            high_bpm=config.heart_rate_warning_high_bpm,
            margin_bpm=config.heart_rate_warning_margin_bpm,
        ):
            log.warning(message)
        try:
            filtered = mdn_filter(
                pixels,
                fs,
                rr_hz,
                hr_bpm / SECONDS_PER_MINUTE,
                notch_half_width_hz=config.mdn_notch_half_width_bpm
                / SECONDS_PER_MINUTE,
                upper_limit_hz=config.mdn_upper_limit_bpm / SECONDS_PER_MINUTE,
                order=config.mdn_order,
            )
            log.info(
                f"MDN cardiac filter with heart rate {hr_bpm:.0f}/min (as entered, "
                "not checked on the data)."
            )
            return filtered, "mdn"
        except ValueError as error:
            log.warning(f"MDN filter not applied ({error}): {not_removed}")
    return moving_average(pixels, fs, config.moving_average_seconds), "moving_average"


def _regional_signals(
    pixels: np.ndarray, roi_masks: dict[str, np.ndarray], log: RunLog
) -> pd.DataFrame:
    """Sum of the pixels of each region; NaN for an empty region, with a warning."""
    signals = {}
    for name, region in roi_masks.items():
        if region.any():
            signals[name] = pixels[:, region].sum(axis=1)
        else:
            signals[name] = np.full(len(pixels), np.nan)
            log.warning(f"Region {name} is empty (no lung pixels): its signal is NaN.")
    return pd.DataFrame(signals)


def _breath_table(
    time: np.ndarray, breaths: np.ndarray, flags: pd.DataFrame
) -> pd.DataFrame:
    """Breath table: times and frames of start, peak and end, then the flags."""
    start, peak, end = breaths.T
    table = pd.DataFrame(
        {
            "breath_id": np.arange(len(breaths)),
            "start_time": time[start],
            "peak_time": time[peak],
            "end_time": time[end],
            "start_idx": start,
            "peak_idx": peak,
            "end_idx": end,
            "method": "eit_global",
        }
    )
    return pd.concat([table, flags], axis=1)


def preprocess(
    raw: ReconstructedFrameData,
    config: PreprocessingConfig | None = None,
    exclusions: Sequence[Segment] = (),
    context: RecordingContext | None = None,
) -> EITRecording:
    """Preprocess one recording: cardiac step, breaths, quality, lung mask, regions.

    Same inputs give the same output, so an application only needs to store the
    inputs (file, config, exclusions, context).

    Args:
        raw:        Parsed recording with pixels (Draeger ``.bin`` or Timpel).
        config:     Parameters; defaults of ``PreprocessingConfig`` if None.
        exclusions: Parts to exclude chosen by the user (``source="manual"``).
        context:    Patient, position, protocol and notes.

    Returns:
        The recording ready for analysis.

    Raises:
        ValueError: if the input is unusable, no breath is found, or no valid breath
            is left for the lung mask.
    """
    config = config if config is not None else PreprocessingConfig()
    context = context if context is not None else RecordingContext()
    log = RunLog(logger)
    _check_input(raw)
    fs = raw.fs
    name = Path(raw.filename).name if raw.filename else "recording"
    log.info(f"fastEIT {__version__}: preprocessing {name}.")
    if raw.vendor != "draeger":
        log.warning(
            "Image orientation and the sign of impedance were verified on Draeger "
            f"recordings only, not on {raw.vendor or 'this vendor'}."
        )

    # 1-2. Time, pixels, device exclusions.
    time = recording_time(raw)
    duration = float(time[-1] + 1 / fs)
    pixels = pixel_values(raw)
    missing = missing_frames(pixels)
    device_segments = _device_exclusions(raw, time, missing, log)
    if missing.any():
        pixels = fill_missing_frames(pixels, missing)

    # 3. Respiratory rate and cardiac step.
    rr_hz = _respiratory_rate(pixels, fs, config, log)
    pixels, cardiac_filter = _cardiac_step(pixels, fs, rr_hz, config, log)

    # 4. Breaths on the global signal of all pixels (the lung mask needs the breaths).
    detection_signal = np.nansum(pixels, axis=(1, 2))
    breaths, near_invalid = detect_breaths(
        detection_signal,
        fs,
        invalid_percentile=config.invalid_data_percentile,
        invalid_multiplier=config.invalid_data_multiplier,
        baseline_seconds=config.breath_baseline_seconds,
        min_duration_seconds=config.breath_min_duration_seconds,
        low_amplitude_fraction=config.breath_low_amplitude_fraction,
        invalid_margin_seconds=config.invalid_data_margin_seconds,
    )
    if len(breaths) == 0:
        raise ValueError("No breaths found: the recording cannot be preprocessed.")

    # 5. Quality flags and exclusions.
    flags = quality_flags(
        detection_signal,
        fs,
        breaths,
        near_invalid,
        short_seconds=config.breath_short_seconds,
        long_seconds=config.breath_long_seconds,
        ti_ratio_low=config.ti_ratio_low,
        ti_ratio_high=config.ti_ratio_high,
        window_breaths=config.local_window_breaths,
        min_breaths=config.local_min_breaths,
        amplitude_ratio_low=config.amplitude_ratio_low,
        amplitude_ratio_high=config.amplitude_ratio_high,
        duration_ratio_high=config.duration_ratio_high,
    )
    table = _breath_table(time, breaths, flags)
    segments = merge_overlapping(
        widen(
            [*exclusions, *device_segments], config.exclusion_margin_seconds, duration
        )
    )
    excluded, labels = breaths_in_segments(table.start_time, table.end_time, segments)
    flagged = flags.any(axis=1).to_numpy()
    table["excluded"] = excluded
    table["exclusion_label"] = pd.Series(labels, dtype="string")
    table["is_valid"] = ~flagged & ~excluded
    log.info(
        f"{len(table)} breaths: {table.is_valid.sum()} valid, {flagged.sum()} with a "
        f"quality flag, {excluded.sum()} excluded."
    )

    # 6. Lung mask, regions, regional signals.
    mask, tiv_map, _ = watershed_mask(
        pixels,
        detection_signal,
        breaths,
        table.is_valid.to_numpy(),
        config.lung_mask_threshold_fraction,
    )
    mask = remove_small_regions(mask, config.lung_mask_min_region_pixels)
    log.info(
        f"Lung mask: {mask.sum()} pixels (watershed at "
        f"{config.lung_mask_threshold_fraction:.0%} of the maximum TIV, from the "
        f"valid breaths)."
    )
    roi_masks = lung_regions(mask)
    roi_signals = _regional_signals(pixels, roi_masks, log)

    # 7. Auxiliary signals (pressures, flow, device markers), cleaned, not filtered.
    aux = auxiliary_signals(raw)
    if len(aux.columns):
        log.info(f"Auxiliary signals with data: {', '.join(aux.columns)}.")
    else:
        log.info("No auxiliary signals with data.")

    return EITRecording(
        raw=raw,
        config=config,
        context=context,
        exclusions=tuple(segments),
        time=time,
        rates={
            "respiratory_rate_hz": rr_hz,
            "heart_rate_hz": (
                config.heart_rate_bpm / SECONDS_PER_MINUTE
                if config.heart_rate_bpm is not None
                else None
            ),
        },
        cardiac_filter=cardiac_filter,
        pixels=pixels.astype(np.float32),
        image_mask=pixels.max(axis=0) > pixels.min(axis=0),
        detection_signal=detection_signal,
        global_signal=pixels[:, mask].sum(axis=1),
        breaths=table,
        lung_mask=mask,
        tiv_map=tiv_map,
        roi_masks=roi_masks,
        roi_signals=roi_signals,
        aux=aux,
        log=tuple(log.messages),
    )
