# fastEIT - Export Layer (tables for R, SPSS and databases)

The export layer (layer 4) takes the clinical indices of layer 3 and writes them as
tables: one analysis at a time, or a whole study at once. The objects of the earlier
layers store each value once, in the table of the layer that computes it; here they
are joined, so that a breath or a PEEP step is one row with everything.

1. [Entry point](#1-entry-point)
2. [Keys: patient, recording, phase](#2-keys-patient-recording-phase)
3. [Tables](#3-tables)
4. [Column names](#4-column-names)
5. [CSV files](#5-csv-files)
6. [Limits](#6-limits)

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

The phase needs a name only when a recording has two phases of the same protocol (e.g.
a baseline and a recovery). `combine_tables()` refuses two analyses with the same three
keys.

The clinical data (side, position, interface, outcome) stay in the case report form and
are joined at the time of the analysis on `patient_id` and `label`: one row per
recording here, one row per patient in R or SPSS (`tidyr::pivot_wider()` on `label`).

## 3. Tables

| Table | One row per | Content |
|-------|-------------|---------|
| `recordings` | analysis | `protocol`, `file`, `vendor`, `fs`, `start_time` and `end_time` of the breaths of the phase, `n_breaths`, `n_used`, typed `tidal_volume_ml`, `cardiac_filter`, `fasteit_version`; the summary of layer 3 in wide format; for a PEEP trial, the global indices of each step by PEEP level of the protocol |
| `breaths` | breath of the phase | times, quality flags, `excluded`, `exclusion_label`, `is_valid`, `used`, PEEP, plateau, driving pressure, measured tidal volume (layer 2); `peep_step`, `peep_set` (layer 2b, PEEP trials); EELI, TIV, GI, centre of ventilation, respiratory rate, tidal volume used, compliance (layer 3); regional values as columns |
| `steps` | PEEP step (PEEP trials) | `step`, measured `peep`, `peep_set`, times, breaths analysed, driving pressure and tidal volume with their sources (layer 2b); medians of the breath indices, compliance and % of the best, collapse, overdistension, changes vs the previous and the first step (layer 3); regional values as columns |
| `settings` | parameter | `section` (`fasteit`, `preprocessing`, `context`, `segmentation`, `analysis`), `parameter`, `value`: with the file, enough to compute the same values again |
| `log` | message | `layer` (`preprocessing`, `segmentation`, `indices`), `message` (warnings start with `WARNING:`) |

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
| `{index}__{region}` | `tiv__ventral`, `compliance_ml__dorsal_left` | regional values in `breaths` and `steps` |
| `{index}__{region}__{statistic}` | `gi__global__median`, `share__dorsal__q1` | summary of a tidal recording in `recordings` |
| `{feature}__{region}` | `crossing_peep__global`, `ri__dorsal` | summary of a PEEP trial in `recordings` |
| `{index}__peep{level}` | `collapse__peep16`, `compliance_ml__peep8` | global indices of each step by PEEP level of the protocol, in `recordings`; a level met twice (up and down) gets `_2` |

Two underscores separate the index from the region, because both can contain one
(`compliance_ml`, `dorsal_left`). Back to one row per region in R:

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
and Python. Files with the same name in the folder are replaced.

## 6. Limits

- The tables are built from the objects of layer 3: the export does not read the files
  or run the pipeline (a study from a file of segments is the next step).
- The images (pixels, lung mask, tidal maps) are not exported.
- `file` is the name of the source file: if it holds names or dates, rename the files
  before sharing the tables.
- `peep_set`, and with it the columns by PEEP level, needs the levels of the protocol
  (`SegmentationConfig.peep_levels`); without them it is empty
  ([`segmentation.md`](segmentation.md)).
