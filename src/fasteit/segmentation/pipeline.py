"""segment_peep_trial(): split a preprocessed PEEP trial into PEEP steps (layer 2b).

Steps (notebooks/06_peep_trial_segmentation.ipynb, section 7):
1. airway pressure: the first channel of ``config.pressure_channels`` in the auxiliary
   signals; otherwise the PEEP and plateau computed by the ventilator (Medibus);
2. pressures of each breath: PEEP, plateau, driving pressure;
3. PEEP steps: placed by hand if given, otherwise found from the PEEP of each breath;
4. breaths analysed in each step, and the step table.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fasteit.config import SegmentationConfig
from fasteit.models.eit_recording import PHASE, EITRecording, Segment
from fasteit.models.peep_trial import PeepTrial
from fasteit.run_log import RunLog
from fasteit.segmentation.breath_pressures import (
    inspiration_onsets,
    plateau_pressures,
    pressure_before,
)
from fasteit.segmentation.peep_steps import (
    analysed_breaths,
    find_steps,
    step_of_each_breath,
    steps_from_segments,
)

logger = logging.getLogger(__name__)

PEEP_TRIAL = "peep_trial"  # label of the PEEP trial phase
# Below this fraction of breaths with a reading, a warning is given.
MIN_FRACTION_READ = 0.5
PRESSURE_UNITS = {"draeger": "mbar", "timpel": "cmH2O"}
STEP_COLUMNS = [
    "step", "peep", "start_time", "end_time", "n_breaths", "n_analysed", "enough",
    "plateau", "driving_pressure", "driving_pressure_source", "n_driving_pressure",
    "enough_driving_pressure",
]  # fmt: skip


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
    recording: EITRecording, config: SegmentationConfig, log: RunLog
) -> tuple[str | None, np.ndarray, np.ndarray]:
    """PEEP and plateau of each breath, and the channel they come from."""
    aux, fs, breaths = recording.aux, recording.fs, recording.breaths
    starts, ends = breaths.start_idx.to_numpy(), breaths.end_idx.to_numpy()

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
        log.info(f"Airway pressure read from {channel}.")
        return channel, peep[:-1], plateau[:-1]

    if _has_data(aux, config.peep_value_channel):
        values = aux[config.peep_value_channel].to_numpy(dtype=float)
        peep = _median_over_breaths(values, starts, ends)
        plateau = np.full(len(breaths), np.nan)
        if _has_data(aux, config.plateau_value_channel):
            values = aux[config.plateau_value_channel].to_numpy(dtype=float)
            plateau = _median_over_breaths(values, starts, ends)
        log.info(
            "No airway pressure waveform: PEEP (and plateau, if there) computed by the "
            f"ventilator, median over each breath ({config.peep_value_channel})."
        )
        return config.peep_value_channel, peep, plateau

    log.warning(
        "No airway pressure in the recording: no PEEP, plateau or driving pressure."
    )
    nothing = np.full(len(breaths), np.nan)
    return None, nothing, nothing.copy()


def _breaths_in_phase(
    starts: np.ndarray, ends: np.ndarray, phase: Segment | None, log: RunLog
) -> np.ndarray:
    """Breaths inside the PEEP trial phase (all breaths if there is no phase).

    A breath cut by the edge of the phase is left out.
    """
    if phase is None:
        return np.ones(len(starts), dtype=bool)
    if phase.category != PHASE or phase.label != PEEP_TRIAL:
        raise ValueError(
            f'The phase must be a Segment with category="{PHASE}" and '
            f'label="{PEEP_TRIAL}", got category="{phase.category}", label="{phase.label}".'
        )
    inside = (starts >= phase.start) & (ends <= phase.end)
    if not inside.any():
        raise ValueError(f"No breath inside the phase {phase.start:g}-{phase.end:g} s.")
    log.info(
        f"PEEP trial phase {phase.start:g}-{phase.end:g} s: {inside.sum()} breaths."
    )
    return inside


def _check_readings(
    peep: np.ndarray, plateau: np.ndarray, in_phase: np.ndarray, log: RunLog
) -> None:
    """Warn when few breaths of the phase have a PEEP or a plateau."""
    with_peep, with_plateau = (
        np.isfinite(peep[in_phase]),
        np.isfinite(plateau[in_phase]),
    )
    log.info(f"PEEP read for {with_peep.sum()} of {in_phase.sum()} breaths.")
    if with_peep.mean() < MIN_FRACTION_READ:
        log.warning("PEEP missing in most breaths: is the pressure signal complete?")
    if with_peep.any() and with_plateau.mean() < MIN_FRACTION_READ:
        log.warning(
            "No end-inspiratory plateau in most breaths: the driving pressure of a step "
            "comes from its breaths with an inspiratory hold, if any (the peak pressure "
            "is not used instead)."
        )


def _automatic_steps(
    peep: np.ndarray,
    in_phase: np.ndarray,
    excluded: np.ndarray,
    valid: np.ndarray,
    starts: np.ndarray,
    config: SegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps from the PEEP of each breath; excluded breaths (maneuvers) do not count."""
    usable_peep = np.where(excluded, np.nan, peep)
    inside = np.flatnonzero(in_phase)
    try:
        step_starts = find_steps(
            usable_peep[inside],
            config.step_threshold,
            config.step_min_breaths,
            config.step_agreement,
        )
    except ValueError as error:
        raise ValueError(f"{error} Place the PEEP steps by hand.") from error
    step = np.full(len(peep), -1, dtype=int)
    step[inside] = step_of_each_breath(step_starts, len(inside))
    levels = np.array(
        [np.nanmedian(usable_peep[step == k]) for k in range(step.max() + 1)]
    )
    # A breath at an intermediate PEEP (e.g. just before a change) is not analysed.
    at_level = np.zeros(len(peep), dtype=bool)
    in_step = step >= 0
    at_level[in_step] = (
        np.abs(peep[in_step] - levels[step[in_step]]) <= config.analysis_peep_tolerance
    )
    analysed = analysed_breaths(
        starts,
        step,
        valid & at_level,
        config.analysis_skip_seconds,
        config.analysis_max_breaths,
    )
    return step, levels, analysed


def _check_manual_levels(
    peep: np.ndarray,
    step: np.ndarray,
    levels: np.ndarray,
    config: SegmentationConfig,
    log: RunLog,
) -> None:
    """Warn when the PEEP typed for a step differs from the PEEP measured in it."""
    for k, level in enumerate(levels):
        measured = peep[step == k]
        if np.isfinite(measured).any() and (
            abs(np.nanmedian(measured) - level) > config.step_threshold
        ):
            log.warning(
                f"Step {k}: PEEP typed {level:g}, measured {np.nanmedian(measured):.1f}."
            )


def _step_driving_pressure(
    in_step: pd.DataFrame,
    analysed: pd.DataFrame,
    level: float,
    typed_plateau: float,
    config: SegmentationConfig,
) -> tuple[float, float, str | None, int, bool]:
    """Plateau, driving pressure, where they come from, how many breaths, and enough.

    In order: the analysed breaths if enough of them have a plateau (continuous
    end-inspiratory pause); otherwise the breaths of the step with an inspiratory hold
    (one is enough; a maneuver excluded by hand is not used); otherwise the value typed
    with a step placed by hand.
    """
    with_driving = analysed[analysed.driving_pressure.notna()]
    if len(with_driving) >= config.analysis_min_breaths:
        source, used = "breaths", with_driving
    else:
        holds = in_step[in_step.driving_pressure.notna() & ~in_step.excluded]
        source, used = "hold", holds
    if len(used):
        enough = source == "hold" or len(used) >= config.analysis_min_breaths
        return (float(used.plateau.median()), float(used.driving_pressure.median()),
                source, len(used), enough)  # fmt: skip
    if np.isfinite(typed_plateau):
        return typed_plateau, typed_plateau - level, "typed", 0, True
    return np.nan, np.nan, None, 0, False


def _steps_table(
    breaths: pd.DataFrame,
    levels: np.ndarray,
    typed_plateaus: np.ndarray,
    config: SegmentationConfig,
    log: RunLog,
) -> pd.DataFrame:
    """One row per step, with its plateau and driving pressure."""
    rows = []
    for k, level in enumerate(levels):
        in_step = breaths[breaths.peep_step == k]
        analysed = in_step[in_step.analysed]
        driving = _step_driving_pressure(
            in_step, analysed, level, typed_plateaus[k], config
        )
        rows.append([
            k, float(level),
            float(in_step.start_time.min()) if len(in_step) else np.nan,
            float(in_step.end_time.max()) if len(in_step) else np.nan,
            len(in_step), len(analysed), len(analysed) >= config.analysis_min_breaths,
            *driving,
        ])  # fmt: skip
        if len(analysed) < config.analysis_min_breaths:
            log.warning(
                f"Step {k} (PEEP {level:.1f}): only {len(analysed)} breaths analysed, "
                f"fewer than {config.analysis_min_breaths}."
            )
    steps = pd.DataFrame(rows, columns=STEP_COLUMNS)
    if not steps.enough_driving_pressure.any():
        log.warning("No step has a driving pressure: compliance cannot be computed.")
    else:
        for k in steps.step[~steps.enough_driving_pressure]:
            log.warning(
                f"Step {k}: no driving pressure (plateau in fewer than "
                f"{config.analysis_min_breaths} analysed breaths, no inspiratory hold, "
                "none typed)."
            )
    return steps


def segment_peep_trial(
    recording: EITRecording,
    config: SegmentationConfig | None = None,
    peep_steps: Sequence[Segment] = (),
    phase: Segment | None = None,
) -> PeepTrial:
    """Split a preprocessed PEEP trial into PEEP steps and choose the breaths analysed.

    A recording can contain more than one protocol, e.g. a tidal baseline and then a PEEP
    trial: ``phase`` limits the segmentation to the PEEP trial. A part that is not
    analysed at all (e.g. a maneuver) is an exclusion of ``preprocess()`` instead: its
    breaths are never analysed and do not count when the steps are found.

    Args:
        recording:  Output of ``preprocess()``.
        config:     Parameters; defaults of ``SegmentationConfig`` if None.
        peep_steps: Steps placed by hand: segments with ``category="peep_step"``, the
                    PEEP level in ``parameters["peep"]`` and optionally the plateau
                    (``parameters["plateau"]``) or the driving pressure
                    (``parameters["driving_pressure"]``) read on the ventilator. If given, they replace the steps found
                    from the pressure, and the breaths analysed are the valid breaths
                    inside them.
        phase:      The PEEP trial part of the recording: a segment with
                    ``category="phase"`` and ``label="peep_trial"``. If None, the whole
                    recording.

    Returns:
        The PEEP trial. Breaths outside the phase or the steps have ``peep_step`` -1.

    Raises:
        ValueError: if there are no steps placed by hand and no pressure to find them,
            a step placed by hand or the phase is not usable, or no breath is inside the
            phase or the steps placed by hand.
    """
    config = config if config is not None else SegmentationConfig()
    log = RunLog(logger)
    b = recording.breaths
    starts, ends = b.start_time.to_numpy(), b.end_time.to_numpy()
    valid, excluded = b.is_valid.to_numpy(), b.excluded.to_numpy()
    in_phase = _breaths_in_phase(starts, ends, phase, log)

    channel, peep, plateau = _breath_pressures(recording, config, log)
    if channel is not None:
        _check_readings(peep, plateau, in_phase, log)

    if peep_steps:
        step, levels, typed_plateaus = steps_from_segments(starts, ends, peep_steps)
        step[~in_phase] = -1
        analysed = (step >= 0) & valid
        source = "manual"
        log.info(f"{len(levels)} PEEP steps placed by hand.")
        _check_manual_levels(peep, step, levels, config, log)
    elif channel is not None:
        step, levels, analysed = _automatic_steps(
            peep, in_phase, excluded, valid, starts, config
        )
        typed_plateaus = np.full(len(levels), np.nan)
        source = "pressure"
        log.info(
            f"{len(levels)} PEEP steps found from the pressure: "
            f"{np.round(levels, 1).tolist()}."
        )
    else:
        raise ValueError(
            "No airway pressure to find the PEEP steps: place them by hand (segments with "
            'category="peep_step" and parameters={"peep": level}).'
        )

    breaths = b.copy()
    breaths["peep"] = peep
    breaths["plateau"] = plateau
    breaths["driving_pressure"] = plateau - peep
    breaths["peep_step"] = step
    breaths["analysed"] = analysed

    return PeepTrial(
        recording=recording,
        config=config,
        pressure_channel=channel,
        pressure_unit=PRESSURE_UNITS.get(recording.raw.vendor),
        source=source,
        steps=_steps_table(breaths, levels, typed_plateaus, config, log),
        breaths=breaths,
        log=tuple(log.messages),
    )
