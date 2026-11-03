"""The results as tables (layer 4): one analysis, a whole study, CSV files.

One analysis is the output of layer 3 for one phase of a recording (``TidalFeatures``
or ``PeepTrialFeatures``). Its tables start every row with the keys ``patient_id``,
``label`` (the recording, e.g. a time point) and ``phase`` (the part analysed):

- ``recordings``: one row per analysis: file, breaths, results of a PEEP trial;
- ``breaths``:    one row per breath of the phase;
- ``steps``:      one row per PEEP step (PEEP trials only);
- ``settings``:   the parameters, the context and the version of fastEIT used;
- ``log``:        the messages of every layer.

The objects of layers 2-3 store each value once; here they are joined, so that a
breath or a step is one row with everything (times, quality, pressures, volumes,
indices). Regional values are columns named ``{index}__{region}``, for the halves of
the lung (ventral, dorsal, right, left; the layers and the quadrants stay in the
objects), with two underscores because both names can contain one: in R
``tidyr::pivot_longer(contains("__"), names_to = c("index", "region"), names_sep = "__")``
gives back one row per region.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fasteit import __version__
from fasteit.features.pipeline import TIDAL
from fasteit.models.eit_recording import EITRecording, Segment
from fasteit.models.features import PeepTrialFeatures, TidalFeatures
from fasteit.segmentation.pipeline import PEEP_TRIAL

KEYS = ["patient_id", "label", "phase"]
TABLES = ["recordings", "breaths", "steps", "settings", "log"]
SEP = "__"  # between index and region (or PEEP level) in the column names
# Regions exported: the halves of the lung. Layers and quadrants stay in the objects.
REGIONS = ["ventral", "dorsal", "right", "left"]

# Frames and detection method stay in the objects: in the tables the times are enough.
BREATH_COLUMNS_LEFT_OUT = ["start_idx", "peak_idx", "end_idx", "method"]


def recording_tables(
    features: TidalFeatures | PeepTrialFeatures,
    label: str | None = None,
    phase: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Tables of one analysis (one phase of a recording).

    Args:
        features: Output of ``tidal_features()`` or ``peep_trial_features()``.
        label:    Name of the recording; ``RecordingContext.label`` if None.
        phase:    Name of the phase, e.g. ``"baseline"``; if None, the protocol
                  (``"tidal"`` or ``"peep_trial"``). Needed when a recording has two
                  phases of the same protocol.

    Returns:
        ``recordings``, ``breaths``, ``steps`` (PEEP trials only), ``settings`` and
        ``log``; every row starts with ``patient_id``, ``label``, ``phase``.
    """
    if isinstance(features, PeepTrialFeatures):
        protocol = PEEP_TRIAL
    elif isinstance(features, TidalFeatures):
        protocol = TIDAL
    else:
        raise TypeError(
            "features must be the output of tidal_features() or peep_trial_features(), "
            f"got {type(features).__name__}."
        )
    context = _recording_of(features).context
    keys = {
        "patient_id": context.patient_id,
        "label": context.label if label is None else label,
        "phase": protocol if phase is None else phase,
    }

    breaths = _breath_table(features)
    tables = {"recordings": _recording_row(features, breaths, protocol)}
    tables["breaths"] = breaths
    if isinstance(features, PeepTrialFeatures):
        tables["steps"] = _step_table(features)
    tables["settings"] = _settings_table(features)
    tables["log"] = _log_table(features)
    return {name: _with_keys(table, keys) for name, table in tables.items()}


def combine_tables(
    analyses: Sequence[dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """The tables of several analyses (a study) as one set of tables.

    A table missing in an analysis (``steps`` of a tidal recording) is skipped; columns
    missing in an analysis (e.g. the results of a PEEP trial in a tidal row) are empty.

    Args:
        analyses: Outputs of ``recording_tables()``.

    Returns:
        The same tables, with the rows of every analysis.

    Raises:
        ValueError: if there is no analysis, or two have the same ``patient_id``,
            ``label`` and ``phase``.
    """
    if not analyses:
        raise ValueError("No analysis to combine.")
    combined = {}
    for name in TABLES:
        parts = [tables[name] for tables in analyses if name in tables]
        if parts:
            combined[name] = pd.concat(parts, ignore_index=True)
    repeated = combined["recordings"][combined["recordings"].duplicated(KEYS)]
    if len(repeated):
        first = repeated.iloc[0]
        raise ValueError(
            f"Two analyses with patient_id={first.patient_id!r}, label={first.label!r}, "
            f"phase={first.phase!r}: give each recording its own label, and each "
            "phase of a recording its own name."
        )
    return combined


def write_csv(tables: dict[str, pd.DataFrame], folder: str | Path) -> dict[str, Path]:
    """Write one CSV file per table, e.g. ``folder/breaths.csv``.

    UTF-8, comma separated, decimal point, empty cell for a missing value, booleans as
    1 / 0: readable by R, SPSS, Excel and Python. Files with the same name are replaced.

    Args:
        tables: Output of ``recording_tables()`` or ``combine_tables()``.
        folder: Folder of the files, created if missing.

    Returns:
        The path of each file, by table name.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        path = folder / f"{name}.csv"
        _booleans_as_numbers(table).to_csv(path, index=False, encoding="utf-8")
        paths[name] = path
    return paths


# --- one analysis ---


def _recording_of(features: TidalFeatures | PeepTrialFeatures) -> EITRecording:
    """The preprocessed recording (layer 2) of an analysis."""
    if isinstance(features, PeepTrialFeatures):
        return features.trial.recording
    return features.recording


def _with_keys(table: pd.DataFrame, keys: dict[str, str]) -> pd.DataFrame:
    """The table with the keys as its first columns."""
    table = table.copy()
    for position, (name, value) in enumerate(keys.items()):
        table.insert(position, name, value)
    return table


def _regions_as_columns(regions: pd.DataFrame, key: str) -> pd.DataFrame:
    """One row per ``key`` (breath or step), one column ``{index}__{region}`` per index
    and exported region (``REGIONS``)."""
    indices = [c for c in regions.columns if c not in (key, "region")]
    regions = regions[regions.region.isin(REGIONS)]
    wide = regions.pivot(index=key, columns="region", values=indices)
    # pivot sorts the regions by name: back to the order of REGIONS.
    wide = wide.reindex(columns=pd.MultiIndex.from_product([indices, REGIONS]))
    wide.columns = [f"{index}{SEP}{region}" for index, region in wide.columns]
    return wide.reset_index()


def _in_phase(breaths: pd.DataFrame, phase: Segment | None) -> pd.Series:
    """Breaths inside the phase, as in layer 2b: a breath cut by its edges is out."""
    if phase is None:
        return pd.Series(True, index=breaths.index)
    return (breaths.start_time >= phase.start) & (breaths.end_time <= phase.end)


def _breath_table(features: TidalFeatures | PeepTrialFeatures) -> pd.DataFrame:
    """One row per breath of the phase: layers 2 (and 2b) and 3 joined on
    ``breath_id``, with ``used`` (the breath entered the indices of the phase)."""
    if isinstance(features, PeepTrialFeatures):
        trial = features.trial
        measured = trial.breaths.copy()  # layer 2 plus peep_step and analysed
        measured = measured[_in_phase(measured, trial.phase)]
        used = measured.pop("analysed")
        # Level of the protocol of the step of each breath (none outside the steps).
        level_of_step = trial.steps.set_index("step").peep_set
        measured.insert(
            measured.columns.get_loc("peep_step") + 1,
            "peep_set",
            measured.peep_step.map(level_of_step),
        )
        indices = features.breaths
    else:
        measured = features.recording.breaths.copy()
        in_phase = features.breaths.breath_id[features.breaths.in_phase]
        measured = measured[measured.breath_id.isin(in_phase)]
        used = measured.is_valid  # tidal: every valid breath of the phase
        indices = features.breaths.drop(columns="in_phase")
    measured = measured.drop(columns=BREATH_COLUMNS_LEFT_OUT)
    measured.insert(measured.columns.get_loc("is_valid") + 1, "used", used)

    table = measured.merge(indices, on="breath_id", how="left", validate="one_to_one")
    regional = _regions_as_columns(features.regions, "breath_id")
    table = table.merge(regional, on="breath_id", how="left", validate="one_to_one")
    return table.reset_index(drop=True)


def _step_table(features: PeepTrialFeatures) -> pd.DataFrame:
    """One row per step: layers 2b (level, breaths, driving pressure, tidal volume) and
    3 (indices, Costa, changes) joined on ``step``, regions as columns."""
    measured = features.trial.steps
    indices = features.steps.drop(columns="peep")  # the level is in layer 2b
    table = measured.merge(indices, on="step", how="left", validate="one_to_one")
    regional = _regions_as_columns(features.step_regions, "step")
    return table.merge(regional, on="step", how="left", validate="one_to_one")


def _trial_results(features: PeepTrialFeatures) -> dict[str, float]:
    """The results of a PEEP trial, one column ``{feature}__{region}`` each: crossing
    PEEP, best PEEP, R/I and recruited volume (global and by exported region).

    Only what needs the methods of layer 3; the medians of the breaths of a tidal
    recording and the steps by PEEP level are summaries of the ``breaths`` and
    ``steps`` tables, made at analysis time.
    """
    columns = {}
    for row in features.summary.itertuples(index=False):
        if row.region == "global" or row.region in REGIONS:
            columns[f"{row.feature}{SEP}{row.region}"] = row.value
    return columns


def _recording_row(
    features: TidalFeatures | PeepTrialFeatures, breaths: pd.DataFrame, protocol: str
) -> pd.DataFrame:
    """One row: the file, the breaths of the phase, the results of a PEEP trial."""
    recording = _recording_of(features)
    row = {
        "protocol": protocol,
        "file": Path(recording.raw.filename).name,
        "vendor": recording.raw.vendor,
        "fs": recording.fs,
        "start_time": breaths.start_time.min(),  # of the breaths of the phase
        "end_time": breaths.end_time.max(),
        "n_breaths": len(breaths),
        "n_used": int(breaths.used.sum()),
        "tidal_volume_ml": recording.context.tidal_volume_ml,  # typed, if any
        "cardiac_filter": recording.cardiac_filter,
        "fasteit_version": __version__,
    }
    if isinstance(features, PeepTrialFeatures):
        row.update(_trial_results(features))
    return pd.DataFrame([row])


def _settings_table(features: TidalFeatures | PeepTrialFeatures) -> pd.DataFrame:
    """One row per parameter of every layer, with the context and the version: with
    the file, enough to compute the same values again."""
    recording = _recording_of(features)
    sections = [("preprocessing", recording.config), ("context", recording.context)]
    if isinstance(features, PeepTrialFeatures):
        sections.append(("segmentation", features.trial.config))
        sections.append(("analysis", features.config))
    rows = [("fasteit", "version", __version__)]
    for section, settings in sections:
        for parameter, value in dataclasses.asdict(settings).items():
            rows.append((section, parameter, "" if value is None else str(value)))
    return pd.DataFrame(rows, columns=["section", "parameter", "value"])


def _log_table(features: TidalFeatures | PeepTrialFeatures) -> pd.DataFrame:
    """One row per message of every layer, in the order they were written."""
    logs = [("preprocessing", _recording_of(features).log)]
    if isinstance(features, PeepTrialFeatures):
        logs.append(("segmentation", features.trial.log))
    logs.append(("indices", features.log))
    rows = [(layer, message) for layer, messages in logs for message in messages]
    return pd.DataFrame(rows, columns=["layer", "message"])


# --- files ---


def _is_boolean(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _booleans_as_numbers(table: pd.DataFrame) -> pd.DataFrame:
    """Booleans as 1 / 0 (empty if missing), the convention of SPSS."""
    table = table.copy()
    for column in table.columns:
        values = table[column]
        if values.dtype == bool or values.dtype == "boolean":
            table[column] = values.astype("Int8")
        elif values.dtype == object:
            # A boolean column combined with a table without it holds True, False, NaN.
            present = values.dropna()
            if len(present) and present.map(_is_boolean).all():
                numbers = values.map(lambda v: pd.NA if pd.isna(v) else int(v))
                table[column] = numbers.astype("Int8")
    return table
