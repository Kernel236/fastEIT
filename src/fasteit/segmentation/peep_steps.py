"""PEEP steps of a PEEP trial and the breaths analysed in each step.

Steps come from the PEEP of each breath (``EITRecording.breaths``) or are placed by
hand as segments with their PEEP level; steps placed by hand win over the automatic
ones (notebooks/06_peep_trial_segmentation.ipynb, sections 3, 5 and 7).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fasteit.models.eit_recording import PEEP_STEP, Segment

# Breaths in a row needed to open a step, so that a single odd breath (a maneuver)
# cannot (notebooks/06_peep_trial_segmentation.ipynb, section 3).
STEP_MIN_BREATHS = 3


def find_steps(
    peep: np.ndarray, threshold: float, min_breaths: int, agreement: float
) -> np.ndarray:
    """First breath of each PEEP step.

    A new step starts when ``min_breaths`` breaths in a row are further than
    ``threshold`` from the level of the current step and within ``agreement`` of each
    other; the level of the current step is the median PEEP of its breaths so far (of
    the first ``min_breaths`` breaths at the start). With ``min_breaths`` of 2 or more a
    single odd breath (a maneuver) cannot open a step. A breath without PEEP (NaN) takes
    the PEEP of the next breath that has one.

    Args:
        peep:        PEEP of each breath.
        threshold:   Half of the step of the protocol (e.g. 1 for steps of 2).
        min_breaths: Breaths in a row needed to open a step.
        agreement:   Largest spread of those breaths.

    Returns:
        Index of the first breath of each step; the first is always 0.

    Raises:
        ValueError: if no breath has a PEEP.
    """
    peep = _fill_backward(np.asarray(peep, dtype=float))
    step_starts = [0]
    level = np.median(peep[:min_breaths])  # level of the first step: its first breaths
    breath = 1
    while breath < len(peep):
        next_breaths = peep[breath : breath + min_breaths]
        opens_step = (
            len(next_breaths) == min_breaths  # enough breaths left
            and np.all(
                np.abs(next_breaths - level) > threshold
            )  # all far from the level
            and np.ptp(next_breaths) <= agreement  # and close to each other
        )
        if opens_step:
            step_starts.append(breath)
            level = np.median(next_breaths)
            breath += min_breaths
        else:
            # The breath stays in the current step: its level is the median of the
            # breaths of the step so far, at least its first min_breaths, so that one
            # odd breath cannot move it.
            step_start = step_starts[-1]
            level_until = max(breath + 1, step_start + min_breaths)
            level = np.median(peep[step_start:level_until])
            breath += 1
    return np.array(step_starts, dtype=int)


def _fill_backward(values: np.ndarray) -> np.ndarray:
    """NaN replaced with the next value that is not NaN (the last ones with the last)."""
    if np.isnan(values).all():
        raise ValueError("No breath has a PEEP: the steps cannot be found.")
    # bfill: each NaN takes the next value; ffill: NaN at the end take the last value.
    return pd.Series(values).bfill().ffill().to_numpy()


def step_of_each_breath(step_starts: np.ndarray, n_breaths: int) -> np.ndarray:
    """Step number of each breath, from the first breath of each step.

    A breath belongs to the last step that starts at or before it: searchsorted counts
    the steps that start at or before each breath, minus one gives the step number.

    Args:
        step_starts: Index of the first breath of each step, increasing.
        n_breaths: Number of breaths.

    Returns:
        The step of each breath; -1 before the first step.
    """
    breaths = np.arange(n_breaths)
    return np.searchsorted(step_starts, breaths, side="right") - 1


def protocol_levels(
    levels: np.ndarray, protocol: Sequence[float] | None, tolerance: float
) -> np.ndarray:
    """PEEP level of the protocol of each step.

    The measured PEEP of every step can be off the set level by about the same amount
    (mbar to cmH2O, PEEP valve: +0.6 cmH2O in every step of a trial of notebook 06, 1.1
    cmH2O at PEEP 0, nearer to 2 than to 0). That offset, the median distance of the
    steps from their nearest level, is removed first; then each step gets the nearest
    level. A step further than ``tolerance`` from every level (a level that is not in
    the protocol) gets NaN. An offset of half a step or more cannot be told apart from
    the next level.

    Args:
        levels:    PEEP of each step, measured or typed (cmH2O).
        protocol:  PEEP levels of the protocol, or None.
        tolerance: Largest distance from a level of the protocol (cmH2O).

    Returns:
        One level per step; NaN where no level of the protocol is near enough, and for
        every step without the levels of the protocol.
    """
    levels = np.asarray(levels, dtype=float)
    if protocol is None or len(levels) == 0:
        return np.full(len(levels), np.nan)
    candidates = np.asarray(protocol, dtype=float)

    def nearest(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Distance of each step (rows) from each level of the protocol (columns).
        distance = np.abs(values[:, None] - candidates[None, :])
        return candidates[np.argmin(distance, axis=1)], distance.min(axis=1)

    first_guess, first_distance = nearest(levels)
    # Offset common to every step, from the steps near a level: a step outside the
    # protocol does not move it.
    near = first_distance <= tolerance
    offset = np.median(levels[near] - first_guess[near]) if near.any() else 0.0
    level, distance = nearest(levels - offset)
    return np.where(distance <= tolerance, level, np.nan)


def steps_from_segments(
    breath_starts: np.ndarray, breath_ends: np.ndarray, segments: Sequence[Segment]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Steps placed by hand: the breaths inside each segment, its PEEP and plateau.

    A breath belongs to a segment when it starts and ends inside it: a breath cut by
    the edge of a segment is left out. Segments are numbered in time order. The plateau,
    or directly the driving pressure, is optional (e.g. read on the ventilator during an
    inspiratory hold), when the pressure curve is missing or not reliable; so is the
    tidal volume of the step (e.g. its mean on the case report form). A value of None
    is not typed.

    Args:
        breath_starts: Start time of each breath, in seconds.
        breath_ends:   End time of each breath, in seconds.
        segments: Segments with ``category="peep_step"``, the PEEP level in
            ``parameters["peep"]`` (cmH2O) and optionally ``parameters["plateau"]`` or
            ``parameters["driving_pressure"]`` (the plateau wins if both are given) and
            ``parameters["tidal_volume"]`` (mL).

    Returns:
        ``step``: step number of each breath (-1 outside every segment);
        ``levels``: PEEP level of each step; ``plateaus``: plateau typed for each step,
        or PEEP + driving pressure typed (NaN if neither is given); ``volumes``: tidal
        volume typed for each step (NaN if not given).

    Raises:
        ValueError: if a segment is not a PEEP step, has no PEEP level, overlaps
            another one or has a typed driving pressure that is not positive, or no
            breath is inside any segment.
    """
    ordered = sorted(segments, key=lambda s: s.start)
    for segment in ordered:
        if segment.category != PEEP_STEP or "peep" not in segment.parameters:
            raise ValueError(
                f"Segment {segment.start:g}-{segment.end:g} s is not a PEEP step: use "
                f'category="{PEEP_STEP}" and parameters={{"peep": level}}.'
            )
    for before, after in zip(ordered[:-1], ordered[1:], strict=True):
        if after.start < before.end:
            raise ValueError(
                f"PEEP steps {before.start:g}-{before.end:g} s and "
                f"{after.start:g}-{after.end:g} s overlap."
            )

    step = np.full(len(breath_starts), -1, dtype=int)
    for k, segment in enumerate(ordered):
        inside = (breath_starts >= segment.start) & (breath_ends <= segment.end)
        step[inside] = k
    if not (step >= 0).any():
        raise ValueError(
            "No breath is inside the PEEP steps placed by hand (times in s?)."
        )
    levels = np.array([float(s.parameters["peep"]) for s in ordered])
    plateaus = np.array([_typed_plateau(s) for s in ordered])
    volumes = np.array([_typed_value(s.parameters, "tidal_volume") for s in ordered])
    return step, levels, plateaus, volumes


def _typed_value(parameters: dict, key: str) -> float:
    """A value typed in the parameters of a segment; NaN if missing or None."""
    value = parameters.get(key)
    return np.nan if value is None else float(value)


def _typed_plateau(segment: Segment) -> float:
    """The plateau typed for a step, or PEEP + the driving pressure typed; NaN if none.

    Raises:
        ValueError: if the typed driving pressure (plateau - PEEP) is not positive.
    """
    peep = float(segment.parameters["peep"])
    plateau = _typed_value(segment.parameters, "plateau")
    if np.isnan(plateau):
        plateau = peep + _typed_value(segment.parameters, "driving_pressure")
    if plateau <= peep:
        raise ValueError(
            f"PEEP step {segment.start:g}-{segment.end:g} s: typed driving pressure "
            f"{plateau - peep:g} (plateau - PEEP): the driving pressure must be positive."
        )
    return plateau


def analysed_breaths(
    breath_starts: np.ndarray,
    step: np.ndarray,
    valid: np.ndarray,
    skip_seconds: float,
    max_breaths: int,
) -> np.ndarray:
    """Breaths analysed in each automatic step.

    The breaths in the first ``skip_seconds`` of the step (from its first breath) are
    left out, as EELI is still moving; then the last ``max_breaths`` valid breaths of
    the step are analysed (notebook 06, section 5).

    Args:
        breath_starts: Start time of each breath, in seconds.
        step:          Step number of each breath (-1 = no step).
        valid:         ``is_valid`` of each breath.
        skip_seconds:  Time left out after the change of PEEP.
        max_breaths:   Most breaths analysed per step.

    Returns:
        Boolean array, True for the analysed breaths.

    Raises:
        ValueError: if ``max_breaths`` is less than 1.
    """
    if max_breaths < 1:
        raise ValueError("max_breaths must be at least 1.")
    analysed = np.zeros(len(step), dtype=bool)
    for step_number in np.unique(step[step >= 0]):
        in_step = np.flatnonzero(step == step_number)  # breaths of the step, in order
        step_start = breath_starts[in_step[0]]
        after_skip = breath_starts[in_step] >= step_start + skip_seconds
        usable = in_step[after_skip & valid[in_step]]
        last_usable = usable[-max_breaths:]
        analysed[last_usable] = True
    return analysed
