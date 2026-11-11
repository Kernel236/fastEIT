"""Figures to review a recording before trusting its numbers.

- ``plot_segmentation(raw, segments)``: the parts chosen by the user (phases,
  exclusions, PEEP steps placed by hand) on the global signal and the airway pressure
  of the file, before ``preprocess()``;
- ``plot_breaths(result)``: the breaths found by ``preprocess()``, which ones are used
  for the indices, flagged or excluded; for a PEEP trial the steps with their PEEP
  level; the lung mask with its tidal image.

Colours and markers are fixed here, so that every figure shows the same thing in the
same way. Each function returns a matplotlib Figure.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from fasteit.models.eit_recording import (
    EXCLUSION,
    PEEP_STEP,
    PHASE,
    EITRecording,
    Segment,
)
from fasteit.models.features import PeepTrialFeatures, TidalFeatures
from fasteit.models.peep_trial import PeepTrial
from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.preprocessing.auxiliary import auxiliary_signals
from fasteit.preprocessing.pipeline import (
    PRESSURE_CHANNELS,
    global_impedance,
    recording_time,
)

INK = "#2b2b2b"
USED = "#3987e5"  # breaths used for the indices (blue)
FLAGGED = "#e8730c"  # breaths with a quality flag (orange)
EXCLUDED = "#9a9a9a"  # excluded breaths and parts (grey)
PHASE_COLOURS = {"tidal": "#3987e5", "peep_trial": "#e8730c"}
BAND_ALPHA = 0.12


def plot_segmentation(
    raw: ReconstructedFrameData, segments: Sequence[Segment] = (), title: str = ""
) -> Figure:
    """The segments typed by the user on the file, before ``preprocess()``.

    Phases are coloured by protocol, exclusions are grey and hatched, PEEP steps placed
    by hand carry their level. The airway pressure (cmH2O) is drawn below when the file
    has it.

    Args:
        raw:      Output of ``load_data()``.
        segments: Phases, exclusions and PEEP steps (``Segment``), in seconds from the
                  start of the recording.
        title:    Title of the figure.

    Returns:
        The figure: global impedance on top (``global_impedance``, with a gap where a
        frame has missing pixels), airway pressure below.
    """
    time = recording_time(raw)
    signal = global_impedance(raw)
    aux, _ = auxiliary_signals(raw)
    pressure_channel = next((c for c in PRESSURE_CHANNELS if c in aux.columns), None)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(12, 4.5), sharex=True, height_ratios=[3, 1]
    )
    top.plot(time, signal, color=INK, lw=0.8)
    top.set_ylabel("global impedance (a.u.)")
    for segment in segments:
        _draw_segment(top, segment)
    top.legend(handles=_segment_legend(segments), loc="upper right", frameon=False)
    top.set_title(title, loc="left", pad=14)  # room for the names of the segments

    if pressure_channel is not None:
        bottom.plot(time, aux[pressure_channel], color=INK, lw=0.6)
        bottom.set_ylabel("Paw (cmH2O)")
    else:
        _no_data(bottom, "no airway pressure")
    bottom.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def plot_breaths(
    result: EITRecording | PeepTrial | TidalFeatures | PeepTrialFeatures,
    title: str = "",
) -> Figure:
    """The breaths found by ``preprocess()`` and the ones used for the indices.

    Each breath is a marker at its end-inspiratory peak on the signal the breaths were
    found on: used for the indices (filled blue circle), valid but not used (empty
    blue circle, e.g. outside the phase, or in the first seconds of a PEEP step),
    flagged by a quality check (orange triangle), excluded (grey cross, with the
    excluded parts as grey bands). For a PEEP trial the steps are shaded and labelled
    with their PEEP level of the protocol (or the measured level), and the PEEP of each
    breath is drawn below. On the right: the tidal image inside the lung mask.

    Args:
        result: Output of ``preprocess()``, ``segment_peep_trial()``,
                ``tidal_features()`` or ``peep_trial_features()``.
        title:  Title of the figure.

    Returns:
        The figure.
    """
    recording, breaths, used, steps = _breath_review(result)

    fig = plt.figure(figsize=(13, 4.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[3, 1])
    top = fig.add_subplot(grid[0, 0])
    bottom = fig.add_subplot(grid[1, 0], sharex=top)
    image = fig.add_subplot(grid[:, 1])

    # Signal, excluded parts and steps.
    signal = recording.detection_signal
    top.plot(recording.time, signal, color=INK, lw=0.7)
    for part in recording.exclusions:
        top.axvspan(part.start, part.end, color=EXCLUDED, alpha=0.25, hatch="//", lw=0)
    if steps is not None:
        _draw_steps(top, bottom, steps)

    # One marker per breath at its peak.
    peaks = signal[breaths.peak_idx.to_numpy()]
    times = breaths.peak_time.to_numpy()
    excluded = breaths.excluded.to_numpy()
    flagged = ~breaths.is_valid.to_numpy() & ~excluded
    valid_not_used = breaths.is_valid.to_numpy() & ~used
    groups = [
        (used, dict(marker="o", color=USED), "used"),
        (valid_not_used, dict(marker="o", mfc="none", color=USED), "valid, not used"),
        (flagged, dict(marker="^", color=FLAGGED), "flagged"),
        (excluded, dict(marker="x", color=EXCLUDED), "excluded"),
    ]
    for selected, style, name in groups:
        top.plot(
            times[selected], peaks[selected], ls="none", ms=5,
            label=f"{name} ({selected.sum()})", **style,
        )  # fmt: skip
    top.set_ylabel("global impedance (a.u.)")
    top.set_title(title, loc="left")
    top.tick_params(labelbottom=False)

    # PEEP of each breath.
    peep = breaths.peep.to_numpy()
    if np.isfinite(peep).any():
        bottom.plot(times, peep, ls="none", marker=".", ms=3, color=EXCLUDED)
        bottom.plot(times[used], peep[used], ls="none", marker=".", ms=4, color=USED)
        bottom.set_ylabel("PEEP (cmH2O)")
    else:
        _no_data(bottom, "no airway pressure")
    bottom.set_xlabel("time (s)")

    _draw_lung(image, recording)
    # Legend under the lung image, away from the signal.
    handles, labels = top.get_legend_handles_labels()
    image.legend(
        handles, labels, title="breaths", loc="upper center",
        bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8,
    )  # fmt: skip
    fig.tight_layout()
    return fig


# --- helpers ---


def _breath_review(
    result: EITRecording | PeepTrial | TidalFeatures | PeepTrialFeatures,
) -> tuple[EITRecording, pd.DataFrame, np.ndarray, pd.DataFrame | None]:
    """Recording, breath table, breaths used for the indices and steps (or None)."""
    if isinstance(result, PeepTrialFeatures):
        result = result.trial
    if isinstance(result, PeepTrial):
        breaths = result.breaths
        return result.recording, breaths, breaths.analysed.to_numpy(), result.steps
    if isinstance(result, TidalFeatures):
        breaths = result.recording.breaths
        in_phase = breaths.breath_id.isin(
            result.breaths.breath_id[result.breaths.in_phase]
        )
        used = breaths.is_valid.to_numpy() & in_phase.to_numpy()
        return result.recording, breaths, used, None
    if isinstance(result, EITRecording):
        return result, result.breaths, result.breaths.is_valid.to_numpy(), None
    raise TypeError(
        "result must be the output of preprocess(), segment_peep_trial(), "
        f"tidal_features() or peep_trial_features(), got {type(result).__name__}."
    )


def _draw_segment(ax: plt.Axes, segment: Segment) -> None:
    """A band for a segment typed by the user, with its name at the top."""
    position, align = (segment.start + segment.end) / 2, "center"
    if segment.category == PHASE:
        colour = PHASE_COLOURS.get(segment.label, INK)
        ax.axvspan(segment.start, segment.end, color=colour, alpha=BAND_ALPHA, lw=0)
        name = segment.label.replace("_", " ")
        position, align = segment.start, "left"  # long: the name at its start
    elif segment.category == EXCLUSION:
        ax.axvspan(
            segment.start, segment.end, color=EXCLUDED, alpha=0.3, hatch="//", lw=0
        )
        name = segment.label or "excluded"
    elif segment.category == PEEP_STEP:
        ax.axvspan(segment.start, segment.end, color=INK, alpha=0.08, lw=0)
        name = f"PEEP {segment.parameters.get('peep', '?')}"
    else:
        raise ValueError(f"Unknown segment category {segment.category!r}.")
    ax.text(
        position, 1.0, name, transform=ax.get_xaxis_transform(),
        ha=align, va="bottom", fontsize=8, color=INK,
    )  # fmt: skip


def _segment_legend(segments: Sequence[Segment]) -> list[Patch]:
    """One legend entry per kind of segment drawn."""
    entries = []
    for protocol, colour in PHASE_COLOURS.items():
        if any(s.category == PHASE and s.label == protocol for s in segments):
            label = f"phase: {protocol.replace('_', ' ')}"
            entries.append(Patch(color=colour, alpha=0.3, label=label))
    if any(s.category == EXCLUSION for s in segments):
        entries.append(
            Patch(facecolor=EXCLUDED, alpha=0.4, hatch="//", label="excluded")
        )
    if any(s.category == PEEP_STEP for s in segments):
        entries.append(Patch(color=INK, alpha=0.15, label="PEEP step"))
    return entries


def _draw_steps(top: plt.Axes, bottom: plt.Axes, steps: pd.DataFrame) -> None:
    """Steps as alternate bands, labelled with their PEEP level; level lines below."""
    for k, step in enumerate(steps.itertuples(index=False)):
        if k % 2 == 0:
            for ax in (top, bottom):
                ax.axvspan(step.start_time, step.end_time, color=INK, alpha=0.05, lw=0)
        level = step.peep_set if np.isfinite(step.peep_set) else step.peep
        top.text(
            (step.start_time + step.end_time) / 2, 0.98, f"{level:g}",
            transform=top.get_xaxis_transform(), ha="center", va="top", fontsize=8,
        )  # fmt: skip
        bottom.hlines(step.peep, step.start_time, step.end_time, color=INK, lw=1.5)


def _draw_lung(ax: plt.Axes, recording: EITRecording) -> None:
    """Tidal image inside the lung mask; patient right on the left (seen from the
    feet), ventral on top."""
    tidal = np.where(recording.lung_mask, recording.tiv_map, np.nan)
    ax.imshow(tidal, cmap="Blues", interpolation="nearest")
    ax.contour(recording.lung_mask, levels=[0.5], colors=INK, linewidths=1)
    ax.set_title("tidal image, lung mask", fontsize=9)
    ax.set_xticks([0, tidal.shape[1] - 1], ["R", "L"])
    ax.set_yticks([0, tidal.shape[0] - 1], ["ventral", "dorsal"])
    ax.tick_params(length=0)


def _no_data(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color=EXCLUDED)  # fmt: skip
    ax.set_yticks([])
