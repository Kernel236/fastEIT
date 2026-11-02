"""Clinical indices of a tidal recording or of a PEEP trial (layer 3).

Definitions and evidence: notebooks/07_clinical_indices.ipynb (section 5):

- EELI and TIV of a breath: one sample at the frames of the layer 2 breaths, on the
  signal inside the lung mask; EELI at the start of the breath, TIV = peak - start
  (TREND: Frerichs et al., Thorax 2017, DOI: 10.1136/thoraxjnl-2016-208357);
- per breath: TIV, EELI and share by region, GI, centre of ventilation, respiratory
  rate, compliance in EIT units (TIV / driving pressure) and in mL/cmH2O (tidal volume
  / driving pressure); with a tidal volume, the TIV and compliance of each region in mL
  (share x tidal volume);
- tidal recording: median and quartiles over the valid breaths;
- PEEP trial: per step, medians of the analysed breaths, compliance (also as % of the
  best step), Costa collapse and overdistension, changes vs the previous and the
  first step (EELI in tidal volumes and in mL, compliance); for the trial, the crossing
  PEEP, the best PEEP and the recruitment-to-inflation ratio, global and by region.

The tables hold the keys (``breath_id``, ``step``, ``peep``, ``region``) and what this
layer computes; what layers 2 and 2b measured stays in ``recording.breaths``,
``trial.breaths`` and ``trial.steps``, reached from the same object.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fasteit.config import AnalysisConfig
from fasteit.features.breath_indices import (
    centre_of_ventilation,
    global_inhomogeneity,
    tidal_images,
)
from fasteit.features.costa import collapse_and_overdistension, crossing_peep
from fasteit.features.recruitment import recruitment_to_inflation, ri_steps
from fasteit.models.eit_recording import PHASE, EITRecording, Segment
from fasteit.models.features import PeepTrialFeatures, TidalFeatures
from fasteit.models.peep_trial import PeepTrial
from fasteit.run_log import RunLog

logger = logging.getLogger(__name__)

TIDAL = "tidal"  # label of a tidal phase
SECONDS_PER_MINUTE = 60.0
# Indices summarised over the breaths of a tidal recording: global ones (with the
# pressures of layer 2) and regional ones.
GLOBAL_INDICES = [
    "respiratory_rate", "eeli", "tiv", "gi", "cov_vd", "cov_rl", "peep", "plateau",
    "driving_pressure", "tidal_volume_used", "compliance", "compliance_ml",
]  # fmt: skip
REGIONAL_INDICES = ["eeli", "tiv", "share", "tiv_ml", "compliance", "compliance_ml"]
# Indices of a PEEP step: the median of its analysed breaths.
STEP_MEDIANS = ["respiratory_rate", "eeli", "tiv", "gi", "cov_vd", "cov_rl"]
# Changes between steps of the compliance, in points of % of the best step (the EELI
# change is computed apart).
STEP_CHANGES = ["compliance_pct_of_best", "compliance_ml_pct_of_best"]


# --- small helpers ---


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """numerator / denominator, NaN where the denominator is not positive (or NaN)."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    # Divide only where the denominator is positive; elsewhere the result stays NaN.
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _median(values: pd.Series | np.ndarray) -> float:
    """Median of the values that are not NaN (NaN if there are none)."""
    values = np.asarray(values, dtype=float)
    known = values[~np.isnan(values)]
    if len(known) == 0:
        return np.nan
    return float(np.median(known))


def _pct_of_best(values: pd.Series) -> pd.Series:
    """Values as % of the highest one (NaN if there is no positive value)."""
    best = values.max()
    if not best > 0:
        return values * np.nan
    return 100 * values / best


# --- indices of each breath ---


def _breath_indices(
    recording: EITRecording, tidal_volume: np.ndarray, source: np.ndarray
) -> pd.DataFrame:
    """Global indices of each breath, one row per breath.

    Args:
        recording:    The preprocessed recording.
        tidal_volume: Tidal volume used for each breath (mL), NaN if none.
        source:       Where the tidal volume of each breath comes from.
    """
    breath_table = recording.breaths
    start_frame = breath_table.start_idx.to_numpy()
    peak_frame = breath_table.peak_idx.to_numpy()
    lung_signal = recording.global_signal  # sum of the pixels inside the lung mask
    driving_pressure = breath_table.driving_pressure.to_numpy()

    eeli = lung_signal[start_frame]
    tiv = lung_signal[peak_frame] - lung_signal[start_frame]
    duration = (breath_table.end_time - breath_table.start_time).to_numpy()

    # Distribution of ventilation, from the tidal image of each breath.
    images = tidal_images(
        recording.pixels, start_frame, peak_frame, recording.lung_mask
    )
    inhomogeneity = [global_inhomogeneity(image) for image in images]
    centres = np.array([centre_of_ventilation(image) for image in images])

    return pd.DataFrame(
        {
            "breath_id": breath_table.breath_id,
            "respiratory_rate": _ratio(SECONDS_PER_MINUTE, duration),
            "eeli": eeli,
            "tiv": tiv,
            "gi": inhomogeneity,
            "cov_vd": centres[:, 0],
            "cov_rl": centres[:, 1],
            "tidal_volume_used": tidal_volume,
            "tidal_volume_source": source,
            "compliance": _ratio(tiv, driving_pressure),
            "compliance_ml": _ratio(tidal_volume, driving_pressure),
        }
    )


def _region_indices(recording: EITRecording, tidal_volume: np.ndarray) -> pd.DataFrame:
    """Indices of each breath in each region, one row per breath and region.

    The rows of a region are in the order of the breath table. The compliance of a
    region is the sum of the compliances of its pixels (they are in parallel, under the
    same driving pressure), so it is the TIV of the region / driving pressure.
    """
    breath_table = recording.breaths
    start_frame = breath_table.start_idx.to_numpy()
    peak_frame = breath_table.peak_idx.to_numpy()
    lung_signal = recording.global_signal
    lung_tiv = lung_signal[peak_frame] - lung_signal[start_frame]
    driving_pressure = breath_table.driving_pressure.to_numpy()

    tables = []
    for region in recording.roi_signals.columns:
        region_signal = recording.roi_signals[region].to_numpy()
        region_tiv = region_signal[peak_frame] - region_signal[start_frame]
        share = _ratio(region_tiv, lung_tiv)  # fraction of the lung TIV, 0-1
        region_volume = share * tidal_volume  # mL
        tables.append(
            pd.DataFrame(
                {
                    "breath_id": breath_table.breath_id.to_numpy(),
                    "region": region,
                    "eeli": region_signal[start_frame],
                    "tiv": region_tiv,
                    "share": share,
                    "tiv_ml": region_volume,
                    "compliance": _ratio(region_tiv, driving_pressure),
                    "compliance_ml": _ratio(region_volume, driving_pressure),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def _recording_tidal_volume(
    recording: EITRecording, log: RunLog
) -> tuple[np.ndarray, np.ndarray]:
    """Tidal volume of each breath: typed for the recording, otherwise measured.

    Returns:
        The volume (mL, NaN if none) and its source (``"typed_recording"``,
        ``"breaths"`` or None) for each breath.
    """
    n_breaths = len(recording.breaths)
    typed = recording.context.tidal_volume_ml
    if typed is not None:
        log.info(f"Tidal volume typed for the recording: {typed:g} mL.")
        volume = np.full(n_breaths, float(typed))
        source = np.full(n_breaths, "typed_recording", dtype=object)
        return volume, source

    measured = recording.breaths.tidal_volume.to_numpy(dtype=float)
    source = np.full(n_breaths, "breaths", dtype=object)
    source[np.isnan(measured)] = None
    if np.isnan(measured).all():
        log.info("No tidal volume: no values in mL.")
    else:
        log.info(f"Tidal volume measured on {recording.tidal_volume_channel}.")
    return measured, source


# --- tidal recordings ---


def _in_phase(
    breath_table: pd.DataFrame, phase: Segment | None, log: RunLog
) -> np.ndarray:
    """Breaths inside the tidal phase (all if there is no phase); a breath cut by its
    edges is left out."""
    if phase is None:
        return np.ones(len(breath_table), dtype=bool)
    if phase.category != PHASE or phase.label != TIDAL:
        raise ValueError(
            f'The phase must be a Segment with category="{PHASE}" and '
            f'label="{TIDAL}", got category="{phase.category}", label="{phase.label}".'
        )
    starts_inside = breath_table.start_time >= phase.start
    ends_inside = breath_table.end_time <= phase.end
    inside = (starts_inside & ends_inside).to_numpy()
    if not inside.any():
        raise ValueError(f"No breath inside the phase {phase.start:g}-{phase.end:g} s.")
    log.info(f"Tidal phase {phase.start:g}-{phase.end:g} s: {inside.sum()} breaths.")
    return inside


def _quartiles(values: pd.Series) -> dict[str, float]:
    """Median, first and third quartile of the values that are not NaN, and how many."""
    known = values.dropna().to_numpy(dtype=float)
    if len(known) == 0:
        return {"median": np.nan, "q1": np.nan, "q3": np.nan, "n_breaths": 0}
    q1, median, q3 = np.quantile(known, [0.25, 0.5, 0.75])
    return {"median": median, "q1": q1, "q3": q3, "n_breaths": len(known)}


def _summary(
    breaths: pd.DataFrame, regions: pd.DataFrame, used: np.ndarray
) -> pd.DataFrame:
    """Median and quartiles of each index over the breaths used: one row per region
    (``"global"`` for the whole lung) and index."""
    rows = []
    for index in GLOBAL_INDICES:
        values = breaths[index][used]
        rows.append({"region": "global", "feature": index, **_quartiles(values)})
    for region, region_breaths in regions.groupby("region", sort=False):
        # The rows of a region are in the order of the breath table, like ``used``.
        for index in REGIONAL_INDICES:
            values = region_breaths[index][used]
            rows.append({"region": region, "feature": index, **_quartiles(values)})
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
    breath_table = recording.breaths

    volume, source = _recording_tidal_volume(recording, log)
    breaths = _breath_indices(recording, volume, source)
    regions = _region_indices(recording, volume)
    breaths.insert(1, "in_phase", _in_phase(breath_table, phase, log))

    # The summary uses the valid breaths of the phase.
    used = breaths.in_phase.to_numpy() & breath_table.is_valid.to_numpy()
    log.info(f"Summary over {used.sum()} valid breaths.")
    if not used.any():
        log.warning("No valid breath: the summary is empty.")
    elif breath_table.driving_pressure[used].isna().all():
        log.info("No driving pressure: no compliance.")

    # The summary also gives the pressures of layer 2 (PEEP, plateau, driving pressure).
    with_pressures = breaths.copy()
    for pressure in ["peep", "plateau", "driving_pressure"]:
        with_pressures[pressure] = breath_table[pressure].to_numpy()

    return TidalFeatures(
        recording=recording,
        breaths=breaths,
        regions=regions,
        summary=_summary(with_pressures, regions, used),
        log=tuple(log.messages),
    )


# --- PEEP trials ---


def _step_breath_volumes(
    trial: PeepTrial, volume: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A tidal volume typed for a step goes to the breaths of that step."""
    volume, source = volume.copy(), source.copy()
    step_of_breath = trial.breaths.peep_step.to_numpy()
    typed_steps = trial.steps[trial.steps.tidal_volume_source == "typed"]
    for _, step in typed_steps.iterrows():
        in_step = step_of_breath == step.step
        volume[in_step] = step.tidal_volume
        source[in_step] = "typed"
    return volume, source


def _steps_with_breaths(trial: PeepTrial):
    """Each step of the trial (a row of ``trial.steps``) with a boolean mask of its
    analysed breaths."""
    analysed = trial.breaths.analysed.to_numpy()
    step_of_breath = trial.breaths.peep_step.to_numpy()
    for _, step in trial.steps.iterrows():
        yield step, analysed & (step_of_breath == step.step)


def _step_indices(trial: PeepTrial, breaths: pd.DataFrame) -> pd.DataFrame:
    """Indices of each step: medians of its analysed breaths, and compliance.

    Compliance in EIT units = median TIV / driving pressure of the step (it also works
    when the driving pressure comes from one inspiratory hold or is typed); in
    mL/cmH2O = tidal volume / driving pressure of the step.
    """
    rows = []
    for step, analysed in _steps_with_breaths(trial):
        medians = {index: _median(breaths[index][analysed]) for index in STEP_MEDIANS}
        rows.append(
            {
                "step": step.step,
                "peep": step.peep,
                **medians,
                "compliance": _ratio(medians["tiv"], step.driving_pressure).item(),
                "compliance_ml": _ratio(
                    step.tidal_volume, step.driving_pressure
                ).item(),
            }
        )
    steps = pd.DataFrame(rows)
    steps["compliance_pct_of_best"] = _pct_of_best(steps.compliance)
    steps["compliance_ml_pct_of_best"] = _pct_of_best(steps.compliance_ml)
    return steps


def _step_region_indices(trial: PeepTrial, regions: pd.DataFrame) -> pd.DataFrame:
    """Indices of each step in each region, from its analysed breaths."""
    # The rows of each region are in the order of the breath table.
    region_tables = {
        name: table for name, table in regions.groupby("region", sort=False)
    }
    rows = []
    for step, analysed in _steps_with_breaths(trial):
        for region, region_breaths in region_tables.items():
            analysed_breaths = region_breaths[analysed]
            share = _median(analysed_breaths.share)
            tiv = _median(analysed_breaths.tiv)
            volume = share * step.tidal_volume  # mL
            rows.append(
                {
                    "step": step.step,
                    "region": region,
                    "eeli": _median(analysed_breaths.eeli),
                    "tiv": tiv,
                    "share": share,
                    "tiv_ml": volume,
                    "compliance": _ratio(tiv, step.driving_pressure).item(),
                    "compliance_ml": _ratio(volume, step.driving_pressure).item(),
                }
            )
    return pd.DataFrame(rows)


def _step_pixel_compliance(trial: PeepTrial) -> np.ndarray:
    """Compliance of each pixel at each step: median tidal image of the analysed
    breaths / driving pressure of the step (NaN for a step without them).

    Returns:
        (n_steps, rows, cols), NaN outside the lung mask.
    """
    recording = trial.recording
    images = tidal_images(
        recording.pixels,
        recording.breaths.start_idx.to_numpy(),
        recording.breaths.peak_idx.to_numpy(),
        recording.lung_mask,
    )
    per_step = []
    for step, analysed in _steps_with_breaths(trial):
        if analysed.any() and step.driving_pressure > 0:
            median_image = np.median(images[analysed], axis=0)
            per_step.append(median_image / step.driving_pressure)
        else:
            per_step.append(np.full(recording.lung_mask.shape, np.nan))
    return np.array(per_step)


def _eeli_change(
    steps: pd.DataFrame,
    reference: pd.DataFrame | pd.Series,
    volume: pd.Series,
    reference_volume: pd.Series | float,
) -> tuple[pd.Series, pd.Series]:
    """Change of EELI of each step from a reference step, in tidal volumes and in mL.

    Converted with the TIV and the tidal volume of the step at the lower PEEP of the two,
    as in Wang et al., Med Sci Monit 2020;26:e922609, DOI: 10.12659/MSM.922609.
    """
    this_step_is_lower = steps.peep <= reference.peep
    tiv_low = steps.tiv.where(this_step_is_lower, reference.tiv)
    volume_low = volume.where(this_step_is_lower, reference_volume)
    in_tidal_volumes = (steps.eeli - reference.eeli) / tiv_low
    return in_tidal_volumes, in_tidal_volumes * volume_low


def _add_changes(steps: pd.DataFrame, tidal_volume: np.ndarray) -> pd.DataFrame:
    """Changes vs the previous step and vs the first step.

    EELI in tidal volumes and in mL (``_eeli_change``); compliance in points of % of the
    best step.

    Args:
        steps:        One row per step, with ``peep``, ``eeli``, ``tiv`` and the indices.
        tidal_volume: Tidal volume of each step (mL), NaN if none.
    """
    previous = steps.shift(1)  # the row of the step before (NaN for the first step)
    first = steps.iloc[0]
    volume = pd.Series(tidal_volume, index=steps.index)

    tiv_vs_previous, ml_vs_previous = _eeli_change(
        steps, previous, volume, volume.shift(1)
    )
    tiv_vs_first, ml_vs_first = _eeli_change(steps, first, volume, volume.iloc[0])
    changes = {
        "d_eeli_tiv_vs_previous": tiv_vs_previous,
        "d_eeli_tiv_vs_first": tiv_vs_first,
        "d_eeli_ml_vs_previous": ml_vs_previous,
        "d_eeli_ml_vs_first": ml_vs_first,
    }
    for index in STEP_CHANGES:
        changes[f"d_{index}_vs_previous"] = steps[index] - previous[index]
        changes[f"d_{index}_vs_first"] = steps[index] - first[index]
    return pd.concat([steps, pd.DataFrame(changes)], axis=1)


def _recruitment_rows(
    trial: PeepTrial,
    steps: pd.DataFrame,
    step_regions: pd.DataFrame,
    config: AnalysisConfig,
    log: RunLog,
) -> list[dict]:
    """R/I and recruited volume between the steps nearest to the two PEEP levels of the
    config, globally and for each region (rows of the summary)."""
    pair = ri_steps(
        steps.peep.to_numpy(),
        config.ri_peep_high,
        config.ri_peep_low,
        config.ri_min_peep_drop,
    )
    if pair is None:
        log.warning(
            f"No two steps near PEEP {config.ri_peep_high:g} and {config.ri_peep_low:g} "
            f"with a drop of at least {config.ri_min_peep_drop:g}: no R/I."
        )
        return []
    high, low = pair
    peep_high, peep_low = steps.peep[high], steps.peep[low]
    driving_low = trial.steps.driving_pressure[low]
    volume_low = trial.steps.tidal_volume[low]
    log.info(f"R/I between the steps at PEEP {peep_high:.1f} and {peep_low:.1f}.")

    rows = [
        {"region": "global", "feature": "ri_peep_high", "value": peep_high},
        {"region": "global", "feature": "ri_peep_low", "value": peep_low},
    ]
    ri, recruited = recruitment_to_inflation(
        steps.eeli[high], steps.eeli[low], steps.tiv[low], driving_low,
        peep_high - peep_low, volume_low,
    )  # fmt: skip
    rows += [
        {"region": "global", "feature": "ri", "value": ri},
        {"region": "global", "feature": "recruited_volume_ml", "value": recruited},
    ]
    for region, region_steps in step_regions.groupby("region", sort=False):
        at_high = region_steps[region_steps.step == steps.step[high]].iloc[0]
        at_low = region_steps[region_steps.step == steps.step[low]].iloc[0]
        ri, recruited = recruitment_to_inflation(
            at_high.eeli, at_low.eeli, at_low.tiv, driving_low,
            peep_high - peep_low, at_low.share * volume_low,
        )  # fmt: skip
        rows += [
            {"region": region, "feature": "ri", "value": ri},
            {"region": region, "feature": "recruited_volume_ml", "value": recruited},
        ]
    return rows


def _best_peep(steps: pd.DataFrame, column: str) -> float:
    """PEEP of the step with the highest value of a column (NaN if none has one)."""
    values = steps[column]
    if values.isna().all():
        return np.nan
    return float(steps.peep[values.idxmax()])


def _check_steps(
    steps: pd.DataFrame, trial: PeepTrial, crossing: float, log: RunLog
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

    few = trial.steps.step[~trial.steps.enough.astype(bool)].tolist()
    if few:
        log.warning(
            f"Steps {few} have fewer than {trial.config.analysis_min_breaths} analysed "
            "breaths: they enter the best step, the % of the best and Costa (column "
            "enough of trial.steps to filter them)."
        )

    if np.isnan(steps.eeli.iloc[0]) or np.isnan(steps.tiv.iloc[0]):
        log.warning("The first step has no EELI or TIV: the changes vs it are NaN.")

    if not np.isnan(crossing):
        log.info(f"Collapse and overdistension cross at PEEP {crossing:.1f}.")
    elif steps.collapse.notna().sum() >= 2:
        log.warning("Collapse and overdistension do not cross in the PEEP range.")

    if trial.steps.tidal_volume.isna().all():
        log.info("No tidal volume: no values in mL.")


def peep_trial_features(
    trial: PeepTrial, config: AnalysisConfig | None = None
) -> PeepTrialFeatures:
    """Clinical indices of each breath and of each PEEP step of a PEEP trial.

    The indices of a step come from its analysed breaths (``segment_peep_trial()``):
    medians of the breath indices; compliance in EIT units = median TIV / driving
    pressure of the step, in mL/cmH2O = tidal volume / driving pressure of the step
    (both kept: the EIT compliance also sees what changes in the slice); Costa
    collapse and overdistension from the median tidal image of each step; the
    recruitment-to-inflation ratio between the steps nearest to the PEEP levels of the
    config, globally and by region.

    Args:
        trial:  Output of ``segment_peep_trial()``.
        config: Parameters; defaults of ``AnalysisConfig`` if None.

    Returns:
        The breath, region, step, step-region and summary tables.
    """
    config = config if config is not None else AnalysisConfig()
    log = RunLog(logger)
    recording = trial.recording
    if len(trial.breaths) != len(recording.breaths):
        raise ValueError("The breaths of the trial are not those of its recording.")

    # 1. Indices of each breath, with the tidal volume typed for its step if any.
    volume, source = _recording_tidal_volume(recording, log)
    volume, source = _step_breath_volumes(trial, volume, source)
    breaths = _breath_indices(recording, volume, source)
    regions = _region_indices(recording, volume)

    # 2. Indices of each step, from its analysed breaths.
    steps = _step_indices(trial, breaths)
    step_regions = _step_region_indices(trial, regions)

    # 3. Costa, from the compliance of each pixel at each step.
    costa = collapse_and_overdistension(
        _step_pixel_compliance(trial), steps.peep.to_numpy(), recording.lung_mask
    )
    steps["collapse"] = costa.collapse
    steps["overdistension"] = costa.overdistension

    # 4. Changes between steps, and the summary of the trial.
    steps = _add_changes(steps, trial.steps.tidal_volume.to_numpy(dtype=float))
    crossing = crossing_peep(steps.peep, steps.collapse, steps.overdistension)
    summary_rows = [
        {"region": "global", "feature": "crossing_peep", "value": crossing},
        {"region": "global", "feature": "best_peep_compliance",
         "value": _best_peep(steps, "compliance")},
        {"region": "global", "feature": "best_peep_compliance_ml",
         "value": _best_peep(steps, "compliance_ml")},
    ]  # fmt: skip
    _check_steps(steps, trial, crossing, log)

    # 5. Recruitment-to-inflation ratio, globally and by region.
    summary_rows += _recruitment_rows(trial, steps, step_regions, config, log)
    summary = pd.DataFrame(summary_rows)

    return PeepTrialFeatures(
        trial=trial,
        breaths=breaths,
        regions=regions,
        steps=steps,
        step_regions=step_regions,
        summary=summary,
        log=tuple(log.messages),
    )
