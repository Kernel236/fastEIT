"""Clinical indices of a tidal recording or of a PEEP trial (layer 3).

Definitions and evidence: notebooks/07_clinical_indices.ipynb (section 5):

- EELI and TIV of a breath: one sample at the frames of the layer 2 breaths, on the
  signal inside the lung mask; EELI at the start of the breath, TIV = peak - start
  (TREND: Frerichs et al., Thorax 2017, DOI: 10.1136/thoraxjnl-2016-208357);
- per breath: TIV, EELI and share by region, GI, centre of ventilation, durations,
  compliance in EIT units (TIV / driving pressure) and in mL/cmH2O (tidal volume /
  driving pressure); with a tidal volume, the TIV and compliance of each region in mL
  (share x tidal volume);
- tidal recording: median and quartiles over the valid breaths;
- PEEP trial: per step, medians of the analysed breaths, compliance (also as % of the
  best step), Costa collapse and overdistension, changes vs the previous and the
  first step (EELI in tidal volumes and in mL).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fasteit.features.breath_indices import (
    centre_of_ventilation,
    global_inhomogeneity,
    tidal_images,
)
from fasteit.features.costa import collapse_and_overdistension, crossing_peep
from fasteit.models.eit_recording import PHASE, EITRecording, Segment
from fasteit.models.features import PeepTrialFeatures, TidalFeatures
from fasteit.models.peep_trial import PeepTrial
from fasteit.run_log import RunLog

logger = logging.getLogger(__name__)

TIDAL = "tidal"  # label of a tidal phase
SECONDS_PER_MINUTE = 60.0
# Indices summarised over the breaths, globally and by region.
GLOBAL_INDICES = [
    "duration", "inspiratory_time", "expiratory_time", "ti_ttot", "respiratory_rate",
    "eeli", "tiv", "gi", "cov_vd", "cov_rl", "peep", "plateau", "driving_pressure",
    "tidal_volume", "compliance", "compliance_ml",
]  # fmt: skip
REGIONAL_INDICES = ["eeli", "tiv", "share", "tiv_ml", "compliance", "compliance_ml"]
# Indices of a step: median of its analysed breaths.
STEP_MEDIANS = ["respiratory_rate", "eeli", "tiv", "gi", "cov_vd", "cov_rl"]


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, NaN where the denominator is not positive."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _median(values: pd.Series | np.ndarray) -> float:
    """Median of the values that are not NaN (NaN if there are none)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return float(np.median(values)) if len(values) else np.nan


def _breath_tables(
    recording: EITRecording, tidal_volume: np.ndarray, source: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Indices of each breath (global) and of each breath and region.

    Args:
        recording:    The preprocessed recording.
        tidal_volume: Tidal volume used for each breath (mL), NaN if none.
        source:       Where it comes from, for each breath.
    """
    b = recording.breaths
    start, peak = b.start_idx.to_numpy(), b.peak_idx.to_numpy()
    lung = recording.global_signal
    tiv = lung[peak] - lung[start]
    images = tidal_images(recording.pixels, start, peak, recording.lung_mask)
    centre = np.array([centre_of_ventilation(image) for image in images])
    duration = (b.end_time - b.start_time).to_numpy()
    inspiration = (b.peak_time - b.start_time).to_numpy()
    driving = b.driving_pressure.to_numpy()

    breaths = pd.DataFrame({
        "breath_id": b.breath_id, "start_time": b.start_time, "peak_time": b.peak_time,
        "end_time": b.end_time, "is_valid": b.is_valid, "excluded": b.excluded,
        "duration": duration, "inspiratory_time": inspiration,
        "expiratory_time": duration - inspiration,
        "ti_ttot": _ratio(inspiration, duration),
        "respiratory_rate": _ratio(SECONDS_PER_MINUTE, duration),
        "eeli": lung[start], "tiv": tiv,
        "gi": [global_inhomogeneity(image) for image in images],
        "cov_vd": centre[:, 0], "cov_rl": centre[:, 1],
        "peep": b.peep, "plateau": b.plateau, "driving_pressure": driving,
        "tidal_volume": tidal_volume, "tidal_volume_source": source,
        "compliance": _ratio(tiv, driving),
        "compliance_ml": _ratio(tidal_volume, driving),
    })  # fmt: skip

    regions = []
    for region in recording.roi_signals.columns:
        signal = recording.roi_signals[region].to_numpy()
        region_tiv = signal[peak] - signal[start]
        share = _ratio(region_tiv, tiv)
        regions.append(pd.DataFrame({
            "breath_id": b.breath_id.to_numpy(), "region": region,
            "eeli": signal[start], "tiv": region_tiv, "share": share,
            "tiv_ml": share * tidal_volume,
            "compliance": _ratio(region_tiv, driving),
            "compliance_ml": _ratio(share * tidal_volume, driving),
        }))  # fmt: skip
    return breaths, pd.concat(regions, ignore_index=True)


def _recording_tidal_volume(
    recording: EITRecording, log: RunLog
) -> tuple[np.ndarray, np.ndarray]:
    """Tidal volume of each breath: typed for the recording, otherwise measured."""
    n = len(recording.breaths)
    typed = recording.context.tidal_volume_ml
    if typed is not None:
        log.info(f"Tidal volume typed for the recording: {typed:g} mL.")
        return np.full(n, float(typed)), np.full(n, "typed_recording", dtype=object)
    measured = recording.breaths.tidal_volume.to_numpy(dtype=float)
    if np.isnan(measured).all():
        log.info("No tidal volume: no values in mL.")
        return measured, np.full(n, None, dtype=object)
    log.info(f"Tidal volume measured on {recording.tidal_volume_channel}.")
    return measured, np.where(np.isnan(measured), None, "breaths").astype(object)


def _in_phase(breaths: pd.DataFrame, phase: Segment | None, log: RunLog) -> np.ndarray:
    """Breaths inside the tidal phase (all if there is no phase); a breath cut by its
    edges is left out."""
    if phase is None:
        return np.ones(len(breaths), dtype=bool)
    if phase.category != PHASE or phase.label != TIDAL:
        raise ValueError(
            f'The phase must be a Segment with category="{PHASE}" and '
            f'label="{TIDAL}", got category="{phase.category}", label="{phase.label}".'
        )
    inside = (breaths.start_time >= phase.start) & (breaths.end_time <= phase.end)
    if not inside.any():
        raise ValueError(f"No breath inside the phase {phase.start:g}-{phase.end:g} s.")
    log.info(f"Tidal phase {phase.start:g}-{phase.end:g} s: {inside.sum()} breaths.")
    return inside.to_numpy()


def _summary(
    breaths: pd.DataFrame, regions: pd.DataFrame, used: np.ndarray
) -> pd.DataFrame:
    """Median and quartiles of each index over the breaths used, globally and by
    region (long table)."""
    tables = [("global", breaths[used], GLOBAL_INDICES)]
    # Each region has one row per breath, in the order of the breath table.
    tables += [
        (region, rows[used], REGIONAL_INDICES)
        for region, rows in regions.groupby("region", sort=False)
    ]
    rows = []
    for region, table, indices in tables:
        for index in indices:
            values = table[index].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            q1, median, q3 = (
                np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else [np.nan] * 3
            )
            rows.append({"region": region, "feature": index, "median": median,
                         "q1": q1, "q3": q3, "n_breaths": len(values)})  # fmt: skip
    return pd.DataFrame(rows)


def tidal_features(
    recording: EITRecording, phase: Segment | None = None
) -> TidalFeatures:
    """Clinical indices of each breath of a recording at one setting, and their summary.

    Args:
        recording: Output of ``preprocess()``.
        phase:     The tidal part of the recording: a segment with ``category="phase"``
                   and ``label="tidal"``; the summary uses only its breaths. If None,
                   the whole recording.

    Returns:
        The breath, region and summary tables. The summary uses the valid breaths of
        the phase (no quality flag, not excluded).

    Raises:
        ValueError: if the phase is not usable or no breath is inside it.
    """
    log = RunLog(logger)
    volume, source = _recording_tidal_volume(recording, log)
    breaths, regions = _breath_tables(recording, volume, source)
    breaths["in_phase"] = _in_phase(breaths, phase, log)
    used = (breaths.in_phase & breaths.is_valid).to_numpy()
    log.info(f"Summary over {used.sum()} valid breaths.")
    if not used.any():
        log.warning("No valid breath: the summary is empty.")
    elif breaths.driving_pressure[used].isna().all():
        log.info("No driving pressure: no compliance.")
    return TidalFeatures(
        recording=recording,
        breaths=breaths,
        regions=regions,
        summary=_summary(breaths, regions, used),
        log=tuple(log.messages),
    )


def _step_breath_volumes(
    trial: PeepTrial, volume: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A tidal volume typed for a step goes to the breaths of that step."""
    volume, source = volume.copy(), source.copy()
    step = trial.breaths.peep_step.to_numpy()
    for _, row in trial.steps[trial.steps.tidal_volume_source == "typed"].iterrows():
        volume[step == row.step] = row.tidal_volume
        source[step == row.step] = "typed"
    return volume, source


def _pct_of_best(values: pd.Series) -> pd.Series:
    """Values as % of the highest (NaN if there is no positive value)."""
    best = values.max()
    return 100 * values / best if best > 0 else values * np.nan


def _steps_table(
    trial: PeepTrial, breaths: pd.DataFrame, regions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per step, and one row per step and region, from the analysed breaths."""
    s = trial.steps
    rec = trial.recording
    analysed = breaths.analysed.to_numpy()
    step_of = breaths.peep_step.to_numpy()
    images = tidal_images(
        rec.pixels, rec.breaths.start_idx.to_numpy(), rec.breaths.peak_idx.to_numpy(),
        rec.lung_mask,
    )  # fmt: skip

    # Each region has one row per breath, in the order of the breath table.
    by_region = dict(tuple(regions.groupby("region", sort=False)))

    rows, region_rows, pixel_compliance = [], [], []
    for _, step in s.iterrows():
        use = analysed & (step_of == step.step)
        medians = {index: _median(breaths[index][use]) for index in STEP_MEDIANS}
        driving, volume = step.driving_pressure, step.tidal_volume
        rows.append({
            "step": step.step, "peep": step.peep, "start_time": step.start_time,
            "end_time": step.end_time, "n_breaths": step.n_breaths,
            "n_analysed": int(use.sum()), "enough": step.enough,
            "plateau": step.plateau, "driving_pressure": driving,
            "driving_pressure_source": step.driving_pressure_source,
            "n_driving_pressure": step.n_driving_pressure,
            "enough_driving_pressure": step.enough_driving_pressure,
            "tidal_volume": volume, "tidal_volume_source": step.tidal_volume_source,
            **medians,
            "compliance": _ratio(medians["tiv"], driving).item(),
            "compliance_ml": _ratio(volume, driving).item(),
        })  # fmt: skip
        if use.any() and driving > 0:
            pixel_compliance.append(np.median(images[use], axis=0) / driving)
        else:
            pixel_compliance.append(np.full(rec.lung_mask.shape, np.nan))

        for region, table in by_region.items():
            r = table[use]
            share, tiv = _median(r.share), _median(r.tiv)
            region_rows.append({
                "step": step.step, "region": region, "eeli": _median(r.eeli),
                "tiv": tiv, "share": share, "tiv_ml": share * volume,
                "compliance": _ratio(tiv, driving).item(),
                "compliance_ml": _ratio(share * volume, driving).item(),
            })  # fmt: skip

    steps = pd.DataFrame(rows)
    steps["compliance_pct_of_best"] = _pct_of_best(steps.compliance)
    steps["compliance_ml_pct_of_best"] = _pct_of_best(steps.compliance_ml)
    costa = collapse_and_overdistension(
        np.array(pixel_compliance), steps.peep.to_numpy(), rec.lung_mask
    )
    steps["collapse"], steps["overdistension"] = costa.collapse, costa.overdistension
    return _add_changes(steps), pd.DataFrame(region_rows)


def _add_changes(steps: pd.DataFrame) -> pd.DataFrame:
    """Changes vs the previous step and vs the first step.

    EELI in tidal volumes (change of EELI / TIV of the reference step) and in mL (x the
    tidal volume of the same reference step); compliance in points of % of the best
    step; collapse and overdistension in points of %.
    """
    previous, first = steps.shift(1), steps.iloc[0]
    eeli_tiv_prev = (steps.eeli - previous.eeli) / previous.tiv
    eeli_tiv_first = (steps.eeli - first.eeli) / first.tiv
    changes = {
        "d_eeli_tiv_vs_previous": eeli_tiv_prev,
        "d_eeli_tiv_vs_first": eeli_tiv_first,
        "d_eeli_ml_vs_previous": eeli_tiv_prev * previous.tidal_volume,
        "d_eeli_ml_vs_first": eeli_tiv_first * first.tidal_volume,
    }
    for index in ["compliance_pct_of_best", "compliance_ml_pct_of_best", "collapse",
                  "overdistension"]:  # fmt: skip
        changes[f"d_{index}_vs_previous"] = steps[index] - previous[index]
        changes[f"d_{index}_vs_first"] = steps[index] - first[index]
    return pd.concat([steps, pd.DataFrame(changes)], axis=1)


def _best_peep(steps: pd.DataFrame, column: str) -> float:
    """PEEP of the step with the highest value of a column (NaN if none has one)."""
    values = steps[column]
    return float(steps.peep[values.idxmax()]) if values.notna().any() else np.nan


def _check_steps(
    steps: pd.DataFrame, crossing: float, min_breaths: int, log: RunLog
) -> None:
    """Warn about steps that are missing or weak in what the indices compare."""
    if steps.compliance.isna().all():
        log.warning("No step has a driving pressure: no compliance and no Costa.")
    else:
        missing = steps.step[steps.compliance.isna()].tolist()
        if missing:
            log.warning(
                f"Steps {missing} have no compliance (no driving pressure or no analysed "
                "breaths): they are left out of the best step and of Costa."
            )
    few = steps.step[~steps.enough.astype(bool)].tolist()
    if few:
        log.warning(
            f"Steps {few} have fewer than {min_breaths} analysed breaths: they enter the "
            "best step, the % of the best and Costa (column enough to filter them)."
        )
    if np.isnan(steps.eeli.iloc[0]) or np.isnan(steps.tiv.iloc[0]):
        log.warning("The first step has no EELI or TIV: the changes vs it are NaN.")
    if np.isnan(crossing):
        if steps.collapse.notna().sum() >= 2:
            log.warning("Collapse and overdistension do not cross in the PEEP range.")
    else:
        log.info(f"Collapse and overdistension cross at PEEP {crossing:.1f}.")
    if steps.tidal_volume.isna().all():
        log.info("No tidal volume: no values in mL.")


def peep_trial_features(trial: PeepTrial) -> PeepTrialFeatures:
    """Clinical indices of each breath and of each PEEP step of a PEEP trial.

    The indices of a step come from its analysed breaths (``segment_peep_trial()``):
    medians of the breath indices; compliance in EIT units = median TIV / driving
    pressure of the step, in mL/cmH2O = tidal volume / driving pressure of the step
    (both kept: the EIT compliance also sees what changes in the slice); Costa
    collapse and overdistension from the median tidal image of each step.

    Args:
        trial: Output of ``segment_peep_trial()``.

    Returns:
        The breath, region, step, step-region and summary tables.
    """
    log = RunLog(logger)
    recording = trial.recording
    if len(trial.breaths) != len(recording.breaths):
        raise ValueError("The breaths of the trial are not those of its recording.")
    volume, source = _recording_tidal_volume(recording, log)
    volume, source = _step_breath_volumes(trial, volume, source)
    breaths, regions = _breath_tables(recording, volume, source)
    breaths["peep_step"] = trial.breaths.peep_step.to_numpy()
    breaths["analysed"] = trial.breaths.analysed.to_numpy()

    steps, step_regions = _steps_table(trial, breaths, regions)
    summary = pd.DataFrame([{
        "n_steps": len(steps),
        "crossing_peep": crossing_peep(steps.peep, steps.collapse, steps.overdistension),
        "best_peep_compliance": _best_peep(steps, "compliance"),
        "best_peep_compliance_ml": _best_peep(steps, "compliance_ml"),
    }])  # fmt: skip
    _check_steps(
        steps, summary.crossing_peep[0], trial.config.analysis_min_breaths, log
    )
    return PeepTrialFeatures(
        trial=trial,
        breaths=breaths,
        regions=regions,
        steps=steps,
        step_regions=step_regions,
        summary=summary,
        log=tuple(log.messages),
    )
