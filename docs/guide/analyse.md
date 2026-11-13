# Analyse a recording

One recording, from the file to the tables, with the functions of each step called one
after the other. Every value is in the objects and in the exported tables; nothing has
to be computed by hand.

1. [A tidal recording](#1-a-tidal-recording)
2. [A PEEP trial](#2-a-peep-trial)
3. [A recording with two parts](#3-a-recording-with-two-parts)
4. [Check the result](#4-check-the-result)
5. [Save the tables](#5-save-the-tables)

## 1. A tidal recording

```python
from fasteit.config import PreprocessingConfig
from fasteit.features import tidal_features
from fasteit.models import RecordingContext
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess

raw = load_data("P01_T0.bin")
recording = preprocess(
    raw,
    config=PreprocessingConfig(heart_rate_bpm=85),   # optional: removes the cardiac component
    context=RecordingContext(patient_id="P01", label="T0", tidal_volume_ml=450),
)
tidal = tidal_features(recording)
```

| What | Where |
|------|-------|
| breaths found, with times, quality flags, PEEP, plateau, driving pressure, tidal volume | `recording.breaths` |
| EELI, TIV, GI, centre of ventilation, compliance of each breath | `tidal.breaths` |
| the same by region: ventral, dorsal, right, left (the objects also hold layers and quadrants; the exported tables have the four halves only) | `tidal.regions` |
| median and quartiles over the valid breaths | `tidal.summary` |
| what each step did: checks passed (`OK:`), facts, warnings (`WARNING:`) | `recording.log`, `tidal.log` |

**Optional inputs.** Without `heart_rate_bpm` the cardiac component is not removed (a
0.1 s moving average only). Without a tidal volume (typed, or measured on the ventilator
curves) the values in mL stay empty and compliance is in EIT units only.

## 2. A PEEP trial

```python
from fasteit.config import SegmentationConfig
from fasteit.features import peep_trial_features
from fasteit.models import RecordingContext
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

recording = preprocess(
    load_data("P01_trial.bin"),
    context=RecordingContext(patient_id="P01", label="supine", protocol="peep_trial",
                             tidal_volume_ml=450),
)
trial = segment_peep_trial(
    recording,
    SegmentationConfig(peep_levels=(16, 14, 12, 10, 8, 6, 4, 2, 0)),   # the protocol
)
trial_features = peep_trial_features(trial)
```

| What | Where |
|------|-------|
| PEEP steps: measured level, level of the protocol (`peep_set`), breaths analysed, driving pressure, tidal volume | `trial.steps` |
| breaths with their step and whether they were analysed | `trial.breaths` |
| each step: compliance and % of the best, Costa collapse and overdistension, changes between steps | `trial_features.steps` |
| crossing PEEP, best PEEP, recruitment-to-inflation ratio (global and by region) | `trial_features.summary` |

The steps are found on the airway pressure. Without a pressure signal, or to choose the
breaths yourself, place the steps by hand ([cut a recording](segments.md)). With the
levels of the protocol (`peep_levels`), each step also gets the level of the protocol
(`peep_set`), and crossing PEEP and best PEEP are given on those levels when every step
matches one; otherwise they are given on the PEEP of the steps (measured, or typed with
the steps placed by hand), and the log says which. Without `peep_levels`, `peep_set` is
empty and the PEEP of the steps is used.

## 3. A recording with two parts

A baseline at one PEEP and a PEEP trial in the same file: preprocess the whole file once
(same filters, same lung mask), then analyse each part. Times are in seconds from the
start of the file; [cut a recording](segments.md) shows how to find them.

```python
from fasteit.features import peep_trial_features, tidal_features
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

recording = preprocess(
    load_data("P02_T0.bin"),
    exclusions=[Segment(285, 300, label="movement")],
    context=RecordingContext(patient_id="P02", label="T0", tidal_volume_ml=450),
)
baseline = tidal_features(recording, phase=Segment(0, 290, "phase", "tidal"))
trial = segment_peep_trial(recording, phase=Segment(300, 1500, "phase", "peep_trial"))
trial_features = peep_trial_features(trial)
```

Each part is an analysis of its own. A breath belongs to a part when it starts and ends
inside it. In the exported tables the parts are told apart by the name of the phase
(section 5). The exclusion is widened by 0.5 s on each side: see
[cut a recording](segments.md#rules).

## 4. Check the result

```python
from fasteit.plotting import plot_breaths

plot_breaths(tidal)           # section 1: breaths used, flagged, excluded; lung mask
plot_breaths(trial)           # section 2: the same, with the steps and their level
```

Look at the figure before trusting the numbers: if a stretch of unstable breaths was
analysed, exclude it and preprocess again ([review figures](../review.md)). Then read the
logs (`recording.log`, `trial.log`, `features.log`): a recording with all it needs and no
problem has only checks (`OK:`) and facts; each `WARNING:` says what is missing or wrong
and what it changes.

## 5. Save the tables

```python
from fasteit.export import combine_tables, recording_tables, write_csv

write_csv(recording_tables(tidal), "results/P01_T0/")          # one analysis (section 1)

tables = combine_tables([                                       # two parts (section 3)
    recording_tables(baseline, phase="baseline"),
    recording_tables(trial_features, phase="trial"),
])
write_csv(tables, "results/study/")
```

Every row starts with `patient_id`, `label` and `phase`, so the tables of many recordings
can be put together and joined with the clinical data ([export](../export.md)). Only
Dräger `.bin` and Timpel files can be analysed; `.eit` and `.asc` are read but not
analysed ([supported files](../getting_started.md#supported-files)).

How each step works: [preprocessing](../preprocessing.md),
[segmentation](../segmentation.md), [clinical indices](../features.md).
