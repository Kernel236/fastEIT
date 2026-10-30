"""PEEP steps of a PEEP trial and the breaths analysed in each step.

Steps come from the PEEP of each breath (``breath_pressures``) or are placed by hand as
segments with their PEEP level; steps placed by hand win over the automatic ones
(notebooks/06_peep_trial_segmentation.ipynb, sections 3, 5 and 7).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fasteit.models.eit_recording import Segment

PEEP_STEP = "peep_step"  # Segment.category of a step placed by hand


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
    starts, level, i = [0], np.median(peep[:min_breaths]), 1
    while i < len(peep):
        window = peep[i : i + min_breaths]
        opens_step = (
            len(window) == min_breaths
            and np.all(np.abs(window - level) > threshold)
            and np.ptp(window) <= agreement
        )
        if opens_step:
            starts.append(i)
            level = np.median(window)
            i += min_breaths
        else:
            # At least the first min_breaths breaths of the step, so one odd breath cannot move it.
            level = np.median(peep[starts[-1] : max(i + 1, starts[-1] + min_breaths)])
            i += 1
    return np.array(starts, dtype=int)


def _fill_backward(values: np.ndarray) -> np.ndarray:
    """NaN replaced with the next value that is not NaN (the last ones with the last)."""
    known = np.flatnonzero(~np.isnan(values))
    if len(known) == 0:
        raise ValueError("No breath has a PEEP: the steps cannot be found.")
    position = np.minimum(
        np.searchsorted(known, np.arange(len(values))), len(known) - 1
    )
    return values[known[position]]


def step_of_each_breath(step_starts: np.ndarray, n_breaths: int) -> np.ndarray:
    """Step number of each breath, from the first breath of each step."""
    return np.searchsorted(step_starts, np.arange(n_breaths), side="right") - 1


def steps_from_segments(
    breath_starts: np.ndarray, breath_ends: np.ndarray, segments: Sequence[Segment]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps placed by hand: the breaths inside each segment, its PEEP and plateau.

    A breath belongs to a segment when it starts and ends inside it: a breath cut by
    the edge of a segment is left out. Segments are numbered in time order. The plateau
    is optional (e.g. read on the ventilator), for recordings without a pressure signal.

    Args:
        breath_starts, breath_ends: Start and end time of each breath, in seconds.
        segments: Segments with ``category="peep_step"``, the PEEP level in
            ``parameters["peep"]`` and optionally the plateau in
            ``parameters["plateau"]``.

    Returns:
        ``step``: step number of each breath (-1 outside every segment);
        ``levels``: PEEP level of each step; ``plateaus``: plateau typed for each step
        (NaN if not given).

    Raises:
        ValueError: if a segment is not a PEEP step, has no PEEP level, overlaps
            another one, or no breath is inside any segment.
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
    plateaus = np.array([float(s.parameters.get("plateau", np.nan)) for s in ordered])
    return step, levels, plateaus


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
    for k in np.unique(step[step >= 0]):
        in_step = np.flatnonzero(step == k)
        after_skip = breath_starts[in_step] >= breath_starts[in_step[0]] + skip_seconds
        usable = in_step[after_skip & valid[in_step]]
        analysed[usable[-max_breaths:]] = True
    return analysed
