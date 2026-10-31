"""preprocess(): from the parsed frames to a recording ready for analysis (layer 2).

Fixed order of the steps (notebooks/05_preprocessing_choices.ipynb):
1. time in seconds, pixel copy with the device "no data" values as NaN;
2. frames with missing pixel data filled and excluded;
3. respiratory rate from the spectrum, cardiac step (MDN with a heart rate given by
   the user, otherwise only a moving average);
4. breath detection on the global signal (sum of all pixels);
5. quality flags and exclusions of each breath;
6. lung mask from the valid breaths, regions of interest, regional signals;
7. auxiliary signals (pressures, flow, volume, device markers) with "no data" as NaN,
   in the same units for every vendor (cmH2O, L/s, mL);
8. airway pressures of each breath: PEEP, plateau, driving pressure
   (notebooks/06_peep_trial_segmentation.ipynb);
9. inspiratory tidal volume of each breath, from the ventilator curves.

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
from fasteit.preprocessing.breath_pressures import (
    inspiration_onsets,
    plateau_pressures,
    pressure_before,
)
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
from fasteit.preprocessing.tidal_volume import (
    inspiration_windows,
    volumes_from_flow_curve,
    volumes_from_volume_curve,
)
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


def _has_data(aux: pd.DataFrame, column: str) -> bool:
    return column in aux and aux[column].notna().any()


def _median_over_breaths(
    values: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Median of a per-frame value over each breath (NaN if it is missing there)."""
    out = np.full(len(starts), np.nan)
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        part = values[start:end]
        if len(part) and not np.isnan(part).all():
            out[i] = np.nanmedian(part)
    return out


def _breath_pressures(
    aux: pd.DataFrame,
    fs: float,
    starts: np.ndarray,
    ends: np.ndarray,
    config: PreprocessingConfig,
    log: RunLog,
) -> tuple[str | None, np.ndarray, np.ndarray]:
    """PEEP and plateau of each breath, and the channel they come from.

    From the first airway pressure waveform of ``config.pressure_channels``; without
    one, the PEEP and plateau computed by the ventilator (Medibus), median over each
    breath; otherwise NaN.

    Args:
        aux:    Auxiliary signals (cmH2O).
        fs:     Sampling frequency in Hz.
        starts, ends: First and last frame of each breath.
        config: Parameters.
        log:    Run log.
    """
    channel = next((c for c in config.pressure_channels if _has_data(aux, c)), None)
    if channel is not None:
        paw = aux[channel].to_numpy(dtype=float)
        # The valley that closes the last breath is one more start of inspiration, so
        # that the last breath also has a plateau.
        onsets = inspiration_onsets(paw, fs, np.r_[starts, ends[-1]])
        peep = pressure_before(paw, fs, onsets, config.peep_seconds)
        plateau = plateau_pressures(
            paw, fs, onsets, peep, config.plateau_window_seconds
        )
        log.info(f"PEEP and plateau of each breath read on {channel}.")
        return channel, peep[:-1], plateau[:-1]

    if _has_data(aux, config.peep_value_channel):
        values = aux[config.peep_value_channel].to_numpy(dtype=float)
        peep = _median_over_breaths(values, starts, ends)
        plateau = np.full(len(starts), np.nan)
        if _has_data(aux, config.plateau_value_channel):
            values = aux[config.plateau_value_channel].to_numpy(dtype=float)
            plateau = _median_over_breaths(values, starts, ends)
        log.info(
            "No airway pressure waveform: PEEP (and plateau, if there) computed by the "
            f"ventilator, median over each breath ({config.peep_value_channel})."
        )
        return config.peep_value_channel, peep, plateau

    log.info("No airway pressure: no PEEP, plateau or driving pressure of the breaths.")
    nothing = np.full(len(starts), np.nan)
    return None, nothing, nothing.copy()


def _tidal_volumes(
    aux: pd.DataFrame,
    fs: float,
    table: pd.DataFrame,
    config: PreprocessingConfig,
    log: RunLog,
) -> tuple[str | None, np.ndarray]:
    """Inspiratory tidal volume of each breath (mL), and the channel it comes from.

    From the volume curve; without it, from the integral of the inspiratory flow;
    without both, the tidal volume computed by the ventilator (Medibus), median over
    each breath; otherwise NaN.
    """
    starts, peaks, ends = (
        table[c].to_numpy() for c in ("start_idx", "peak_idx", "end_idx")
    )
    first, after_last = inspiration_windows(starts, peaks, ends, len(aux))
    if _has_data(aux, config.volume_channel):
        volume = aux[config.volume_channel].to_numpy(dtype=float)
        log.info(f"Tidal volume of each breath read on {config.volume_channel}.")
        return config.volume_channel, volumes_from_volume_curve(
            volume, first, after_last
        )
    if _has_data(aux, config.flow_channel):
        flow = aux[config.flow_channel].to_numpy(dtype=float)
        log.info(
            "Tidal volume of each breath from the integral of the inspiratory flow "
            f"({config.flow_channel})."
        )
        return config.flow_channel, volumes_from_flow_curve(flow, fs, first, after_last)
    if _has_data(aux, config.tidal_volume_value_channel):
        values = aux[config.tidal_volume_value_channel].to_numpy(dtype=float)
        log.info(
            "No volume or flow curve: tidal volume computed by the ventilator, median "
            f"over each breath ({config.tidal_volume_value_channel})."
        )
        return config.tidal_volume_value_channel, _median_over_breaths(
            values, starts, ends
        )
    log.info(
        "No volume or flow: no tidal volume of the breaths (it can be typed, "
        "RecordingContext.tidal_volume_ml)."
    )
    return None, np.full(len(table), np.nan)


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
    aux, aux_units = auxiliary_signals(raw)
    if len(aux.columns):
        log.info(
            f"Auxiliary signals with data: {', '.join(aux.columns)} (pressures in cmH2O, "
            "flow in L/s, volumes in mL)."
        )
    else:
        log.info("No auxiliary signals with data.")

    # 8. Airway pressures of each breath (cmH2O).
    pressure_channel, peep, plateau = _breath_pressures(
        aux, fs, table.start_idx.to_numpy(), table.end_idx.to_numpy(), config, log
    )
    table["peep"] = peep
    table["plateau"] = plateau
    table["driving_pressure"] = plateau - peep

    # 9. Inspiratory tidal volume of each breath (mL).
    tidal_volume_channel, tidal_volume = _tidal_volumes(aux, fs, table, config, log)
    table["tidal_volume"] = tidal_volume
    if context.tidal_volume_ml is not None:
        log.info(
            f"Tidal volume typed for the recording: {context.tidal_volume_ml:g} mL."
        )

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
        aux_units=aux_units,
        pressure_channel=pressure_channel,
        tidal_volume_channel=tidal_volume_channel,
        log=tuple(log.messages),
    )
