# fastEIT - Preprocessing Layer

The preprocessing layer (layer 2) takes one parsed recording with pixels and returns
an `EITRecording` ready for breath-by-breath analysis. It is the same for every
protocol (tidal breathing, PEEP trial); the segmentation by protocol and the clinical
indices (EELI, TIV, GI, CoV, compliance) come in the next layers.

The evidence for every choice is in `notebooks/05_preprocessing_choices.ipynb`
(11 real recordings). The steps reimplement
[eitprocessing](https://github.com/EIT-ALIVE/eitprocessing) v1.8.8 (Somhorst et al.,
*JOSS* 2026;11(117):8179, DOI: 10.21105/joss.08179) and the tests check them against
it.

1. [Entry point](#1-entry-point)
2. [Steps](#2-steps)
3. [Output](#3-output)
4. [Logs](#4-logs)
5. [Limits](#5-limits)
6. [File layout](#6-file-layout)

---

## 1. Entry point

```python
from fasteit.config import PreprocessingConfig
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess

raw = load_data("patient01.bin")
recording = preprocess(
    raw,                                             # never modified
    config=PreprocessingConfig(heart_rate_bpm=85),   # every parameter, with its source
    exclusions=[Segment(120, 180, label="cough")],   # parts chosen by the user
    context=RecordingContext(patient_id="PZ01", protocol="tidal"),
)
```

The same inputs always give the same output, so an application only needs to store
the inputs (file, config, exclusions, context) and can recompute the rest.
`PreprocessingConfig` is frozen: to change a parameter make a new one, e.g.
`dataclasses.replace(config, heart_rate_bpm=80)`.

## 2. Steps

| # | Step | Module | Source |
|---|------|--------|--------|
| 1 | Time in seconds, pixels copied with the Dräger "no data" values as NaN | `pipeline` | - |
| 2 | Device exclusions: frames with missing pixels (filled with the nearest frame), timing errors, gaps in the frame times | `exclusions` | - |
| 3 | Respiratory rate from the Welch spectrum | `cardiac_filter` | eitprocessing `RateDetection` |
| 4 | Cardiac step: MDN filter if the user gives the heart rate, otherwise a 0.1 s moving average (the heart is **not** removed) | `cardiac_filter` | Wisse et al., *Physiol Meas* 2024, DOI: 10.1088/1361-6579/ad46e3 |
| 5 | Breath detection on the sum of all pixels: valley, peak, next valley | `breaths` | eitprocessing `BreathDetection` |
| 6 | Quality flags and exclusions of each breath; `is_valid` = no flag and not excluded | `breath_quality`, `exclusions` | notebook §8 |
| 7 | Lung mask from the valid breaths: watershed of the TIV and amplitude maps at 15 % of the maximum TIV, islands < 10 pixels removed | `lung_mask` | eitprocessing `WatershedLungspace`, `FilterROIBySize` |
| 8 | Regions of interest inside the mask and their signals | `regions` | notebook §9 |

Breaths are **flagged, never removed**: the clinician decides what to use.

Quality flags of each breath:

| Column | Rule |
|--------|------|
| `flag_short`, `flag_long` | duration below 0.5 s or above 20 s |
| `flag_suspect_ti_ratio` | Ti/Ttot (from impedance) outside 0.2-0.6 |
| `flag_amplitude_outlier` | amplitude below 0.5× or above 2× the median of the 21 surrounding breaths |
| `flag_duration_outlier` | duration above 2× the median of the 21 surrounding breaths |
| `flag_invalid_data` | within 0.5 s of a gross outlier of the signal |

## 3. Output

`EITRecording` fields:

| Field | Content |
|-------|---------|
| `raw`, `config`, `context` | the inputs |
| `exclusions` | manual and device exclusions, widened by 0.5 s and merged |
| `time` | seconds from the start of the recording |
| `rates`, `cardiac_filter` | respiratory rate, heart rate as entered, filter actually applied (`"mdn"` or `"moving_average"`) |
| `pixels` | (N, 32, 32) float32 after the cardiac step |
| `image_mask` | pixels with a signal, wider than the lung |
| `detection_signal` | sum of all pixels: the breaths are its valleys and peaks |
| `global_signal` | sum of the pixels inside the lung mask |
| `breaths` | one row per breath: times and frames, flags, `excluded`, `exclusion_label`, `is_valid` |
| `lung_mask`, `tiv_map` | ventilated lung and the mean TIV map it comes from |
| `roi_masks`, `roi_signals` | `ventral`, `dorsal`, `right`, `left`, `layer_1`-`layer_4`, four quadrants |
| `log` | what each step did, with warnings |

Images are seen from the feet, like a CT slice: row 0 is ventral, column 0 is the
patient's right.

## 4. Logs

There are two kinds of record, kept apart on purpose:

- **`EITRecording.log`** belongs to the result: it says which filter ran, what was
  excluded and every warning, and is saved with the data.
- **Technical logs** go through Python `logging` (logger names
  `fasteit.preprocessing.*`). The library never configures logging: the script or
  application decides where the messages go, e.g. `logging.basicConfig(level=logging.INFO)`
  in a notebook, or stdout collected by Docker in an application.

## 5. Limits

- Validated as equivalent to eitprocessing, not against a gold standard; 946 breaths
  from two patients and one volunteer, no PEEP trial yet.
- Breath timing comes from impedance: the valley is the minimum of lung volume, not
  the ventilator trigger; patient-ventilator asynchrony is not detected; breaths
  within about 15 s of a sudden EELI step are unreliable.
- Without a heart rate the cardiac component is not removed.
- The lung mask is the ventilated lung. The regions are cut on its vertical extent,
  so they are relative to the ventilated lung: indices of a shift of ventilation
  (centre of ventilation, silent spaces) must use image coordinates.
- Orientation and signal sign are verified on Dräger PulmoVista recordings only.
- Adults only (respiratory rate 6-60/min).

## 6. File layout

```
src/fasteit/
├── config.py                  PreprocessingConfig: every parameter, with its source
├── models/eit_recording.py    Segment, RecordingContext, EITRecording
└── preprocessing/
    ├── cardiac_filter.py      respiratory rate, MDN, moving average, heart rate warnings
    ├── exclusions.py          device and manual exclusions
    ├── breaths.py             breath detection
    ├── breath_quality.py      quality flags
    ├── lung_mask.py           watershed lung mask
    ├── regions.py             regions of interest
    └── pipeline.py            preprocess()
```

*Research use only - not a medical device.*
