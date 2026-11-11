# Export for statistical analysis

The export (layer 4) takes the results of an analysis ([analyse a recording](guide/analyse.md))
and writes them as tables: one analysis at a time, or a whole study at once. In the
analysis objects each value is stored once, in the table of the step that computes it;
here the tables are joined, so that a breath or a PEEP step is one row with everything.
Only recordings that can be analysed (Dräger `.bin`, Timpel) have tables.

1. [Entry point](#1-entry-point)
2. [Keys: patient, recording, phase](#2-keys-patient-recording-phase)
3. [Tables](#3-tables)
4. [Column names](#4-column-names)
5. [CSV files](#5-csv-files)

---

## 1. Entry point

```python
from fasteit.export import combine_tables, recording_tables, write_csv
from fasteit.features import peep_trial_features, tidal_features
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

context = RecordingContext(patient_id="P01", label="T0", tidal_volume_ml=450)
recording = preprocess(load_data("P01_T0.bin"), context=context)

# A baseline, then a PEEP trial, in the same recording: one analysis per phase.
baseline = tidal_features(recording, phase=Segment(0, 290, "phase", "tidal"))
trial = peep_trial_features(
    segment_peep_trial(recording, phase=Segment(320, 1500, "phase", "peep_trial"))
)

tables = combine_tables([
    recording_tables(baseline, phase="baseline"),
    recording_tables(trial, phase="trial"),
])
write_csv(tables, "results/")   # results/recordings.csv, breaths.csv, steps.csv, ...
```

The tables on real recordings (11 tidal recordings, four PEEP trials) are in
`notebooks/08_export.ipynb`.

A study is the same call with the analyses of every recording: one or a hundred, the
tables are the same. For a simple tidal recording `recording_tables(tidal_features(recording))`
is enough.

## 2. Keys: patient, recording, phase

Every row starts with three keys:

| Key | What | Where it comes from |
|-----|------|---------------------|
| `patient_id` | the patient | `RecordingContext.patient_id` |
| `label` | the recording for that patient, e.g. a time point (`T0`) or a position | `RecordingContext.label`, or `recording_tables(..., label="T1")` |
| `phase` | the part of the recording analysed | `recording_tables(..., phase="baseline")`; by default the protocol, `tidal` or `peep_trial` |

By default the phase is the protocol, so a tidal baseline and a PEEP trial of the same
recording are told apart already; two parts of the same protocol (e.g. a baseline and a
recovery) need a name. Naming every part (`baseline`, `trial`) makes the tables easier to
read: see [cut a recording](guide/segments.md).
`combine_tables()` refuses two analyses with the same three keys.

The clinical data (side, position, interface, outcome) stay in the case report form and
are joined at the time of the analysis on `patient_id` and `label`: one row per
recording here, one row per patient in R or SPSS (`tidyr::pivot_wider()` on `label`).

## 3. Tables

| Table | One row per | Content |
|-------|-------------|---------|
| `recordings` | analysis | `protocol`, `file`, `vendor`, `fs`, `start_time` and `end_time` of the breaths of the phase, `n_breaths`, `n_used`, typed `tidal_volume_ml`, `cardiac_filter`, `fasteit_version`; for a PEEP trial its results: crossing PEEP, best PEEP, R/I and recruited volume (global and by half) |
| `breaths` | breath of the phase | times, quality flags, `excluded`, `exclusion_label`, `is_valid`, `used`, PEEP, plateau, driving pressure, measured tidal volume (preprocessing); `peep_step`, `peep_set` (PEEP trials); EELI, TIV, GI, centre of ventilation, respiratory rate, tidal volume used, compliance (clinical indices); regional values as columns |
| `steps` | PEEP step (PEEP trials) | `step`, measured `peep`, `peep_set`, times, breaths analysed, driving pressure and tidal volume with their sources (segmentation); medians of the breath indices, compliance and % of the best, collapse, overdistension, changes vs the previous and the first step (clinical indices); regional values as columns |
| `settings` | parameter | `section` (`fasteit`, `preprocessing`, `context`, `segmentation`, `analysis`), `parameter`, `value`: with the file, enough to compute the same values again |
| `log` | message | `layer` (`preprocessing`, `segmentation`, `indices`), `level` (`ok` for a check that passed, `info`, `warning`), `message` |

**Each value once.** The tables hold what was measured (one row per breath and per
step) and the results of the PEEP-trial methods (Costa collapse and overdistension in
`steps`; crossing PEEP, best PEEP and recruitment-to-inflation ratio, R/I, in
`recordings`, one value per trial). Summaries of these rows are made
at analysis time, with the choices of the study (median or mean, which breaths):

```r
breaths |> filter(used == 1) |>
  group_by(patient_id, label, phase) |>
  summarise(across(c(tiv, gi, share__dorsal, compliance_ml), median))   # one row per recording
steps |> pivot_wider(id_cols = c(patient_id, label), names_from = peep_set,
                     values_from = collapse, names_prefix = "collapse_peep")   # one column per level
```

In SPSS the same are `AGGREGATE` (breaths to one row per recording) and `CASESTOVARS`
(steps to one column per PEEP level).

**Breaths of the phase.** Each analysis exports the breaths of its own phase, all of
them (also the excluded ones and those between steps), so that in a recording with two
phases no breath is exported twice. `used` marks the breaths that entered the indices:
the valid breaths of a tidal phase, the breaths analysed in each step of a PEEP trial.

**Steps.** The step table is the curve of the trial: with `peep_set` on the x axis it
gives the collapse and overdistension curves (Costa) and the crossing PEEP, and the
regional compliance at each PEEP.

```r
library(tidyverse)
steps <- read_csv("results/steps.csv")
ggplot(steps, aes(peep_set)) +
  geom_line(aes(y = collapse)) + geom_line(aes(y = overdistension)) +
  facet_wrap(~ patient_id + label)
```

## 4. Column names

| Pattern | Example | Where |
|---------|---------|-------|
| `{index}__{region}` | `tiv__ventral`, `compliance_ml__left` | regional values in `breaths` and `steps` |
| `{feature}__{region}` | `crossing_peep__global`, `ri__dorsal` | results of a PEEP trial in `recordings` |

The regions exported are the halves of the lung: `ventral`, `dorsal`, `right`, `left`
(patient right and left); the layers and the quadrants stay in the analysis objects
(`features.regions`) and are not exported. Two underscores separate the index from the region, because an
index can contain one (`compliance_ml`). Back to one row per region in R:

```r
steps |> pivot_longer(contains("__"), names_to = c("index", "region"), names_sep = "__")
```

Names are in snake case, start with a letter and are shorter than 64 characters, as
SPSS requires. Units: pressures in cmH2O, volumes in mL, compliance in EIT units per
cmH2O (`compliance`) and mL/cmH2O (`compliance_ml`), times in seconds from the start
of the file.

## 5. CSV files

`write_csv(tables, folder)` writes one file per table: UTF-8, comma separated,
decimal point, an empty cell for a missing value, booleans as 1 / 0. The files are
read without options by R (`readr::read_csv()`), SPSS (File > Import Data > CSV), Excel
and Python. Files with the same name in the folder are replaced. The column `file` is the
name of the source file: rename files whose names hold names or dates before sharing the
tables.

