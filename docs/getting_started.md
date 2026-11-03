# Getting started

## Installation

```bash
pip install git+https://github.com/Kernel236/fastEIT.git
```

For development (tests, linting, the eitprocessing reference) and to build this site:

```bash
git clone https://github.com/Kernel236/fastEIT.git
cd fastEIT
pip install -e ".[dev,docs]"
mkdocs serve        # the site at http://127.0.0.1:8000
```

## A tidal recording

```python
from fasteit.config import PreprocessingConfig
from fasteit.features import tidal_features
from fasteit.models import RecordingContext
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess

raw = load_data("recording.bin")                       # layer 1
recording = preprocess(                                # layer 2
    raw,
    config=PreprocessingConfig(heart_rate_bpm=85),     # optional: MDN cardiac filter
    context=RecordingContext(tidal_volume_ml=450),     # optional: values in mL
)
features = tidal_features(recording)                   # layer 3

features.breaths     # one row per breath: EELI, TIV, GI, centre of ventilation, compliance
features.regions     # one row per breath and region
features.summary     # median and quartiles over the valid breaths
print("\n".join(recording.log))                        # what each step did
```

## A PEEP trial

```python
from fasteit.features import peep_trial_features
from fasteit.segmentation import segment_peep_trial

recording = preprocess(load_data("peep_trial.bin"),
                       context=RecordingContext(tidal_volume_ml=450))
trial = segment_peep_trial(recording)                  # layer 2b: PEEP steps
features = peep_trial_features(trial)                  # layer 3

features.steps          # one row per step: compliance, Costa, changes between steps
features.step_regions   # one row per step and region
features.summary        # crossing PEEP, best PEEP, recruitment-to-inflation ratio
trial.steps             # PEEP level, breaths analysed, driving pressure, tidal volume
```

Without a pressure signal, or to choose the breaths yourself, place the steps by hand
([segmentation](segmentation.md)).

## Tables for R and SPSS

```python
from fasteit.export import combine_tables, recording_tables, write_csv

tables = combine_tables([recording_tables(f) for f in all_features])  # one or many
write_csv(tables, "results/")   # recordings.csv, breaths.csv, steps.csv, settings.csv, log.csv
```

Every row starts with `patient_id`, `label` (the recording) and `phase`; see
[export](export.md).

## Look at the breaths used

```python
from fasteit.plotting import plot_breaths

plot_breaths(trial)   # breaths used, flagged, excluded; steps; lung mask
```

See [review figures](review.md).

## Where the values are

Each value is stored once, in the table of the layer that produces it, and the tables are
joined on `breath_id`, `step` and `region`: see the [data model](data_model.md).
