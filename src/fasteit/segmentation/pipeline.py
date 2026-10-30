"""segment_peep_trial(): split a preprocessed PEEP trial into PEEP steps (layer 2b).

Steps (notebooks/06_peep_trial_segmentation.ipynb, section 7):
1. airway pressure: the first channel of ``config.pressure_channels`` in the auxiliary
   signals; otherwise the PEEP and plateau computed by the ventilator (Medibus);
2. pressures of each breath: PEEP, plateau, driving pressure;
3. PEEP steps: placed by hand if given, otherwise found from the PEEP of each breath;
4. breaths analysed in each step.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fasteit.config import SegmentationConfig
from fasteit.models.eit_recording import EITRecording, Segment
from fasteit.models.peep_trial import PeepTrial
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


class _Log:
    """Collects the messages of one call and sends them to the logger."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)
        logger.info(message)

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")
        logger.warning(message)


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
    recording: EITRecording, config: SegmentationConfig, log: _Log
) -> tuple[str | None, np.ndarray, np.ndarray]:
    """PEEP and plateau of each breath, and the channel they come from."""
    aux, fs, breaths = recording.aux, recording.fs, recording.breaths
    starts, ends = breaths.start_idx.to_numpy(), breaths.end_idx.to_numpy()
    nothing = np.full(len(breaths), np.nan)

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
        peep, plateau = peep[:-1], plateau[:-1]
        log.info(f"Airway pressure read from {channel}.")
        if np.isnan(plateau).mean() > 0.5:
            log.warning(
                "No end-inspiratory plateau in most breaths: plateau and driving "
                "pressure are missing (the peak pressure is not used instead)."
            )
        return channel, peep, plateau

    if _has_data(aux, config.peep_value_channel):
        peep = _median_over_breaths(
            aux[config.peep_value_channel].to_numpy(dtype=float), starts, ends
        )
        plateau = nothing.copy()
        if _has_data(aux, config.plateau_value_channel):
            plateau = _median_over_breaths(
                aux[config.plateau_value_channel].to_numpy(dtype=float), starts, ends
            )
        log.info(
            f"No airway pressure waveform: PEEP and plateau computed by the ventilator "
            f"({config.peep_value_channel}, {config.plateau_value_channel})."
        )
        return config.peep_value_channel, peep, plateau

    log.warning(
        "No airway pressure in the recording: no PEEP, plateau or driving pressure."
    )
    return None, nothing, nothing.copy()


def segment_peep_trial(
    recording: EITRecording,
    config: SegmentationConfig | None = None,
    peep_steps: Sequence[Segment] = (),
) -> PeepTrial:
    """Split a preprocessed PEEP trial into PEEP steps and choose the breaths analysed.

    Args:
        recording:  Output of ``preprocess()``.
        config:     Parameters; defaults of ``SegmentationConfig`` if None.
        peep_steps: Steps placed by hand: segments with ``category="peep_step"`` and the
                    PEEP level in ``parameters["peep"]``. If given, they replace the
                    steps found from the pressure, and the breaths analysed are the valid
                    breaths inside them.

    Returns:
        The PEEP trial.

    Raises:
        ValueError: if there are no steps placed by hand and no pressure to find them,
            or a step placed by hand is not usable.
    """
    config = config if config is not None else SegmentationConfig()
    log = _Log()
    b = recording.breaths
    times, ends = b.start_time.to_numpy(), b.end_time.to_numpy()
    valid = b.is_valid.to_numpy()

    channel, peep, plateau = _breath_pressures(recording, config, log)

    if peep_steps:
        step, levels = steps_from_segments(times, ends, peep_steps)
        analysed = (step >= 0) & valid
        source = "manual"
        log.info(f"{len(levels)} PEEP steps placed by hand.")
    elif channel is not None:
        step = step_of_each_breath(
            find_steps(
                peep,
                config.step_threshold,
                config.step_min_breaths,
                config.step_agreement,
            ),
            len(b),
        )
        levels = np.array(
            [np.nanmedian(peep[step == k]) for k in range(step.max() + 1)]
        )
        analysed = analysed_breaths(
            times,
            step,
            valid,
            config.analysis_skip_seconds,
            config.analysis_max_breaths,
        )
        source = "pressure"
        log.info(
            f"{len(levels)} PEEP steps found from the pressure: {np.round(levels, 1).tolist()}."
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

    rows = []
    for k, level in enumerate(levels):
        in_step = breaths[breaths.peep_step == k]
        n_analysed = int(in_step.analysed.sum())
        rows.append(
            {
                "step": k,
                "peep": float(level),
                "start_time": float(in_step.start_time.min())
                if len(in_step)
                else np.nan,
                "end_time": float(in_step.end_time.max()) if len(in_step) else np.nan,
                "breaths": len(in_step),
                "analysed": n_analysed,
                "enough": n_analysed >= config.analysis_min_breaths,
            }
        )
        if n_analysed < config.analysis_min_breaths:
            log.warning(
                f"Step {k} (PEEP {level:.1f}): only {n_analysed} breaths analysed, fewer "
                f"than {config.analysis_min_breaths}."
            )
    steps = pd.DataFrame(
        rows,
        columns=[
            "step",
            "peep",
            "start_time",
            "end_time",
            "breaths",
            "analysed",
            "enough",
        ],
    )

    return PeepTrial(
        recording=recording,
        config=config,
        pressure_channel=channel,
        source=source,
        steps=steps,
        breaths=breaths,
        log=tuple(log.messages),
    )
