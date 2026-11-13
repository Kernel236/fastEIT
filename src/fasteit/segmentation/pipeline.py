"""segment_peep_trial(): split a preprocessed PEEP trial into PEEP steps (layer 2b).

Steps (notebooks/06_peep_trial_segmentation.ipynb, section 7):
1. breaths of the PEEP trial phase, with the PEEP and plateau read in layer 2
   (``EITRecording.breaths``);
2. PEEP steps: placed by hand if given, otherwise found from the PEEP of each breath;
3. breaths analysed in each step, and the step table.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fasteit.config import SegmentationConfig
from fasteit.models.eit_recording import PHASE, EITRecording, Segment
from fasteit.models.peep_trial import PeepTrial
from fasteit.preprocessing.tidal_volume import TYPED_VOLUME_RANGE_ML
from fasteit.run_log import RunLog
from fasteit.segmentation.peep_steps import (
    STEP_MIN_BREATHS,
    analysed_breaths,
    find_steps,
    protocol_levels,
    step_of_each_breath,
    steps_from_segments,
)

logger = logging.getLogger(__name__)

PEEP_TRIAL = "peep_trial"  # label of the PEEP trial phase
# Below this fraction of breaths with a reading, a warning is given.
MIN_FRACTION_READ = 0.5
# A typed tidal volume more than 15 % from the one measured on the analysed breaths of
# the step gives a warning (a typing error, or one volume typed for a trial in pressure
# control, where it changes with PEEP): a plausibility check, not a measured accuracy.
TIDAL_VOLUME_AGREEMENT = 0.15
# A typed driving pressure more than 2 cmH2O from the one read on the curve of the step
# gives a warning: the curve may not be reliable (e.g. a damped line, a plateau read on
# the fall of the pressure) or the value mistyped. A plausibility check, not a measured
# accuracy; in cmH2O, not in %, because the driving pressure is a small number.
DRIVING_PRESSURE_AGREEMENT = 2.0
# A plateau and a driving pressure typed for the same step must agree (plateau - PEEP)
# within the rounding of values read on the ventilator (cmH2O).
TYPED_PRESSURE_TOLERANCE = 0.5
STEP_COLUMNS = [
    "step", "peep", "peep_set", "peep_measured", "start_time", "end_time", "n_breaths",
    "n_analysed", "enough", "plateau", "driving_pressure", "driving_pressure_source",
    "n_driving_pressure", "enough_driving_pressure", "driving_pressure_measured",
    "tidal_volume", "tidal_volume_source",
]  # fmt: skip


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
    with_peep = np.isfinite(peep[in_phase])
    with_plateau = np.isfinite(plateau[in_phase])
    read = f"PEEP read for {with_peep.sum()} of {in_phase.sum()} breaths."
    if with_peep.all():
        log.ok(read)
    else:
        log.info(read)
    if with_peep.mean() < MIN_FRACTION_READ:
        log.warning("PEEP missing in most breaths: is the pressure signal complete?")
    if with_peep.any() and with_plateau.mean() < MIN_FRACTION_READ:
        log.warning(
            "No end-inspiratory plateau in most breaths: the driving pressure of a step "
            "is the one typed with it, otherwise from its breaths with an inspiratory "
            "hold, if any (the peak pressure is not used instead)."
        )


def _automatic_steps(
    peep: np.ndarray,
    in_phase: np.ndarray,
    excluded: np.ndarray,
    valid: np.ndarray,
    breath_starts: np.ndarray,
    config: SegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps from the PEEP of each breath; excluded breaths (maneuvers) do not count.

    Returns:
        Step number of each breath (-1 outside every step), PEEP level of each step,
        and the breaths analysed.
    """
    # Excluded breaths do not count when the steps are found.
    usable_peep = peep.copy()
    usable_peep[excluded] = np.nan

    # Steps are found on the breaths of the phase only.
    phase_breaths = np.flatnonzero(in_phase)
    try:
        step_starts = find_steps(
            usable_peep[phase_breaths],
            config.step_threshold,
            STEP_MIN_BREATHS,
            config.step_threshold,  # the breaths that open a step agree within it
        )
    except ValueError as error:
        raise ValueError(f"{error} Place the PEEP steps by hand.") from error
    step = np.full(len(peep), -1, dtype=int)
    step[phase_breaths] = step_of_each_breath(step_starts, len(phase_breaths))

    # Level of each step: median PEEP of its breaths.
    n_steps = step.max() + 1
    levels = np.array(
        [np.nanmedian(usable_peep[step == number]) for number in range(n_steps)]
    )

    # A breath at an intermediate PEEP (e.g. just before a change) is not analysed.
    level_of_breath = np.full(len(peep), np.nan)
    in_a_step = step >= 0
    level_of_breath[in_a_step] = levels[step[in_a_step]]
    # Half of the threshold, so that a breath halfway between two steps is never
    # analysed.
    tolerance = config.step_threshold / 2
    at_level = np.abs(peep - level_of_breath) <= tolerance

    analysed = analysed_breaths(
        breath_starts,
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
    compared, differ = 0, 0
    for k, level in enumerate(levels):
        measured = peep[step == k]
        if not np.isfinite(measured).any():
            continue
        compared += 1
        if abs(np.nanmedian(measured) - level) > config.step_threshold:
            differ += 1
            log.warning(
                f"Step {k}: PEEP typed {level:g}, measured {np.nanmedian(measured):.1f}."
            )
    if compared == len(levels) and not differ:
        log.ok("PEEP typed for each step agrees with the PEEP measured in it.")


def _check_typed_pressures(segments: Sequence[Segment], log: RunLog) -> None:
    """Warn when a plateau and a driving pressure typed for the same step disagree (the
    plateau is used)."""
    for k, segment in enumerate(sorted(segments, key=lambda s: s.start)):
        typed = segment.parameters
        if typed.get("plateau") is None or typed.get("driving_pressure") is None:
            continue
        peep, plateau = float(typed["peep"]), float(typed["plateau"])
        driving = float(typed["driving_pressure"])
        if abs(plateau - peep - driving) > TYPED_PRESSURE_TOLERANCE:
            log.warning(
                f"Step {k}: plateau {plateau:g} and driving pressure {driving:g} typed "
                f"with PEEP {peep:g}: the plateau is used."
            )


def _measured_driving_pressure(
    in_step: pd.DataFrame, analysed: pd.DataFrame, config: SegmentationConfig
) -> tuple[float, float, str | None, int]:
    """Plateau and driving pressure of a step read on the pressure curve, where they
    come from and how many breaths.

    In order: the analysed breaths if enough of them have a plateau (continuous
    end-inspiratory pause); otherwise the breaths of the step with an inspiratory hold
    (one is enough; a maneuver excluded by hand is not used).
    """
    with_plateau = analysed[analysed.driving_pressure.notna()]
    if len(with_plateau) >= config.analysis_min_breaths:
        used, source = with_plateau, "breaths"
    else:
        used = in_step[in_step.driving_pressure.notna() & ~in_step.excluded]
        source = "hold"
    if not len(used):
        return np.nan, np.nan, None, 0
    plateau, driving = used.plateau.median(), used.driving_pressure.median()
    return float(plateau), float(driving), source, len(used)


def _step_driving_pressure(
    in_step: pd.DataFrame,
    analysed: pd.DataFrame,
    level: float,
    typed_plateau: float,
    config: SegmentationConfig,
) -> tuple[float, float, str | None, int, bool, float]:
    """Plateau, driving pressure, where they come from, how many breaths, enough, and
    the driving pressure read on the curve.

    The value typed with a step placed by hand comes first, as for the tidal volume:
    it is typed on purpose, e.g. read on the ventilator because the curve is missing or
    not reliable. Otherwise the curve (``_measured_driving_pressure``).
    """
    plateau, driving, source, n = _measured_driving_pressure(in_step, analysed, config)
    measured = driving
    if np.isfinite(typed_plateau):
        plateau, driving, source, n = typed_plateau, typed_plateau - level, "typed", 0
    return plateau, driving, source, n, source is not None, measured


def _check_typed_driving_pressures(steps: pd.DataFrame, log: RunLog) -> None:
    """Warn when a typed driving pressure is far from the one read on the curve of the
    step (``DRIVING_PRESSURE_AGREEMENT``)."""
    typed = steps.driving_pressure_source == "typed"
    compared = steps[typed & steps.driving_pressure_measured.notna()]
    differ = 0
    for _, step in compared.iterrows():
        if (
            abs(step.driving_pressure - step.driving_pressure_measured)
            > DRIVING_PRESSURE_AGREEMENT
        ):
            differ += 1
            log.warning(
                f"Step {step.step}: driving pressure typed {step.driving_pressure:g}, "
                f"measured {step.driving_pressure_measured:.1f} on the curve: the typed "
                "value is used."
            )
    if len(compared) and not differ:
        log.ok(
            f"Typed driving pressure close to the measured one in {len(compared)} steps."
        )


def _typed_volumes(
    tidal_volumes: Sequence[float | None] | None,
    from_segments: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Tidal volume typed for each step, in the segments or as a list (NaN if not)."""
    volumes = from_segments
    if tidal_volumes is not None:
        if np.isfinite(from_segments).any():
            raise ValueError(
                "Tidal volumes typed both in the PEEP step segments and in "
                "tidal_volumes: give them in one place."
            )
        volumes = np.array([np.nan if v is None else float(v) for v in tidal_volumes])
        if len(volumes) != n_steps:
            raise ValueError(
                f"tidal_volumes has {len(volumes)} values for {n_steps} PEEP steps: "
                "give one per step in time order (None for a step without it)."
            )
    if (volumes <= 0).any() or np.isinf(volumes).any():
        raise ValueError("Typed tidal volumes must be positive and finite (mL).")
    return volumes


def _check_typed_volumes(
    steps: pd.DataFrame,
    breaths: pd.DataFrame,
    plausible: tuple[float, float],
    config: SegmentationConfig,
    log: RunLog,
) -> None:
    """Warn about typed tidal volumes outside the plausible range or far from the
    tidal volume measured on the analysed breaths of the step."""
    low, high = plausible
    typed_for_steps = steps[steps.tidal_volume_source == "typed"]
    outside = 0
    for _, step in typed_for_steps.iterrows():
        if not low <= step.tidal_volume <= high:
            outside += 1
            log.warning(
                f"Step {step.step}: typed tidal volume {step.tidal_volume:g} mL is "
                f"outside {low:g}-{high:g} mL: is it in mL?"
            )
    if len(typed_for_steps) and not outside:
        log.ok("Tidal volume typed for each step in the plausible range.")
    typed = steps.tidal_volume_source.isin(["typed", "typed_recording"])
    compared, differ = 0, 0
    for _, step in steps[typed].iterrows():
        in_step = breaths[(breaths.peep_step == step.step) & breaths.analysed]
        measured = in_step.tidal_volume.dropna()
        if not len(measured):
            continue
        compared += 1
        if abs(step.tidal_volume - measured.median()) > (
            TIDAL_VOLUME_AGREEMENT * measured.median()
        ):
            differ += 1
            log.warning(
                f"Step {step.step}: tidal volume typed {step.tidal_volume:g} mL, "
                f"measured {measured.median():.0f} mL."
            )
    if compared and not differ:
        log.ok(f"Typed tidal volume close to the measured one in {compared} steps.")


def _step_tidal_volume(
    analysed: pd.DataFrame, typed_step: float, typed_recording: float | None
) -> tuple[float, str | None]:
    """Tidal volume of a step (mL) and where it comes from.

    In order: typed for the step; typed for the whole recording
    (``RecordingContext.tidal_volume_ml``, e.g. the volume set in volume control); the
    median of the analysed breaths, measured on the ventilator curves.
    """
    if np.isfinite(typed_step):
        return float(typed_step), "typed"
    if typed_recording is not None:
        return float(typed_recording), "typed_recording"
    measured = analysed.tidal_volume.dropna()
    if len(measured):
        return float(measured.median()), "breaths"
    return np.nan, None


def _steps_table(
    breaths: pd.DataFrame,
    levels: np.ndarray,
    typed_plateaus: np.ndarray,
    typed_volumes: np.ndarray,
    recording_volume: float | None,
    config: SegmentationConfig,
    log: RunLog,
) -> pd.DataFrame:
    """One row per step, with its level of the protocol, PEEP measured (median of its
    breaths), plateau, driving pressure and tidal volume."""
    peep_set = protocol_levels(levels, config.peep_levels, config.step_threshold)
    if config.peep_levels is None:
        log.warning(
            "No PEEP levels of the protocol (SegmentationConfig.peep_levels): "
            "peep_set is empty and the results are given on the PEEP of the steps "
            "(measured, or typed with the steps placed by hand)."
        )
    elif np.isfinite(peep_set).all():
        log.ok(f"Every step on a PEEP level of the protocol: {peep_set.tolist()}.")
    else:
        log.info(f"PEEP level of the protocol of each step: {peep_set.tolist()}.")
    rows = []
    for k, level in enumerate(levels):
        in_step = breaths[breaths.peep_step == k]
        analysed = in_step[in_step.analysed]
        driving = _step_driving_pressure(
            in_step, analysed, level, typed_plateaus[k], config
        )
        volume = _step_tidal_volume(analysed, typed_volumes[k], recording_volume)
        rows.append([
            k, float(level), float(peep_set[k]), float(in_step.peep.median()),
            float(in_step.start_time.min()) if len(in_step) else np.nan,
            float(in_step.end_time.max()) if len(in_step) else np.nan,
            len(in_step), len(analysed), len(analysed) >= config.analysis_min_breaths,
            *driving, *volume,
        ])  # fmt: skip
        if config.peep_levels is not None and np.isnan(peep_set[k]):
            log.warning(
                f"Step {k} (PEEP {level:.1f}): no PEEP level of the protocol within "
                f"{config.step_threshold:g} cmH2O."
            )
        if len(analysed) < config.analysis_min_breaths:
            log.warning(
                f"Step {k} (PEEP {level:.1f}): only {len(analysed)} breaths analysed, "
                f"fewer than {config.analysis_min_breaths}."
            )
    steps = pd.DataFrame(rows, columns=STEP_COLUMNS)
    if steps.enough.all():
        log.ok(
            f"Every step has at least {config.analysis_min_breaths} analysed breaths."
        )
    if steps.enough_driving_pressure.all():
        log.ok("Every step has a driving pressure.")
    elif not steps.enough_driving_pressure.any():
        log.warning("No step has a driving pressure: compliance cannot be computed.")
    else:
        for k in steps.step[~steps.enough_driving_pressure]:
            log.warning(
                f"Step {k}: no driving pressure (plateau in fewer than "
                f"{config.analysis_min_breaths} analysed breaths, no inspiratory hold, "
                "none typed)."
            )
    if steps.tidal_volume.isna().all():
        log.info(
            "No tidal volume for the steps (none typed, no volume or flow curve): "
            "compliance in EIT units only."
        )
    return steps


def segment_peep_trial(
    recording: EITRecording,
    config: SegmentationConfig | None = None,
    peep_steps: Sequence[Segment] = (),
    phase: Segment | None = None,
    tidal_volumes: Sequence[float | None] | None = None,
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
                    PEEP level in ``parameters["peep"]`` (cmH2O) and optionally the
                    plateau (``parameters["plateau"]``) or the driving pressure
                    (``parameters["driving_pressure"]``) read on the ventilator, and
                    the tidal volume (``parameters["tidal_volume"]``, mL). If given,
                    they replace the steps found from the pressure, and the breaths
                    analysed are the valid breaths inside them. A typed plateau or
                    driving pressure is used before the pressure curve, which is
                    still read (``driving_pressure_measured``, with a warning when
                    the two differ by more than ``DRIVING_PRESSURE_AGREEMENT``).
        phase:      The PEEP trial part of the recording: a segment with
                    ``category="phase"`` and ``label="peep_trial"``. If None, the whole
                    recording.
        tidal_volumes: Tidal volume typed for each step (mL), in time order, e.g. the
                    mean of each step on the case report form; None for a step
                    without it. One volume for the whole recording goes in
                    ``RecordingContext.tidal_volume_ml`` instead.

    Returns:
        The PEEP trial. Breaths outside the phase or the steps have ``peep_step`` -1.

    Raises:
        ValueError: if there are no steps placed by hand and no pressure to find them,
            a step placed by hand or the phase is not usable (e.g. a typed driving
            pressure not positive), no breath is inside the phase or the steps placed by
            hand, or ``tidal_volumes`` does not have one positive value per step.
    """
    config = config if config is not None else SegmentationConfig()
    log = RunLog(logger)
    b = recording.breaths
    starts, ends = b.start_time.to_numpy(), b.end_time.to_numpy()
    valid, excluded = b.is_valid.to_numpy(), b.excluded.to_numpy()
    in_phase = _breaths_in_phase(starts, ends, phase, log)

    channel = recording.pressure_channel
    peep, plateau = b.peep.to_numpy(), b.plateau.to_numpy()
    if channel is not None:
        log.info(f"PEEP and plateau of each breath from {channel}.")
        _check_readings(peep, plateau, in_phase, log)
    else:
        # The warning is in the log of the preprocessing, where the pressure is read.
        log.info(
            "No airway pressure in the recording: no PEEP, plateau or driving pressure."
        )

    if peep_steps:
        step, levels, typed_plateaus, segment_volumes = steps_from_segments(
            starts, ends, peep_steps
        )
        step[~in_phase] = -1
        analysed = (step >= 0) & valid
        source = "manual"
        log.info(f"{len(levels)} PEEP steps placed by hand.")
        _check_manual_levels(peep, step, levels, config, log)
        _check_typed_pressures(peep_steps, log)
    elif channel is not None:
        step, levels, analysed = _automatic_steps(
            peep, in_phase, excluded, valid, starts, config
        )
        typed_plateaus = np.full(len(levels), np.nan)
        segment_volumes = np.full(len(levels), np.nan)
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
    breaths["peep_step"] = step
    breaths["analysed"] = analysed
    steps = _steps_table(
        breaths,
        levels,
        typed_plateaus,
        _typed_volumes(tidal_volumes, segment_volumes, len(levels)),
        recording.context.tidal_volume_ml,
        config,
        log,
    )
    _check_typed_driving_pressures(steps, log)
    plausible = TYPED_VOLUME_RANGE_ML
    _check_typed_volumes(steps, breaths, plausible, config, log)

    return PeepTrial(
        recording=recording,
        config=config,
        source=source,
        phase=phase,
        steps=steps,
        breaths=breaths,
        log=tuple(log.messages),
    )
