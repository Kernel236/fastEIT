# fastEIT - Preprocessing Layer

The preprocessing layer (layer 2) takes one parsed recording with pixels and returns
an `EITRecording` ready for breath-by-breath analysis. It is the same for every
protocol (tidal breathing, PEEP trial): each breath gets its EIT timings and quality,
and its ventilator values (PEEP, plateau, driving pressure, tidal volume). The
segmentation by protocol and the clinical indices (EELI, TIV, GI, CoV, compliance)
come in the next layers.

The evidence for every choice is in `notebooks/05_preprocessing_choices.ipynb`
(11 real recordings); for the pressures of each breath, in
`notebooks/06_peep_trial_segmentation.ipynb` (four PEEP trials). The EIT steps
reimplement [eitprocessing](https://github.com/EIT-ALIVE/eitprocessing) v1.8.8
(Somhorst et al., *JOSS* 2026;11(117):8179, DOI: 10.21105/joss.08179) and the tests
check them against it.

1. [Entry point](#1-entry-point)
2. [Steps](#2-steps)
3. [Ventilator values of each breath](#3-ventilator-values-of-each-breath)
4. [Output](#4-output)
5. [Logs](#5-logs)
6. [Limits](#6-limits)
7. [File layout](#7-file-layout)

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
    context=RecordingContext(
        patient_id="PZ01",
        protocol="tidal",
        tidal_volume_ml=450,                         # optional: e.g. set in volume control
    ),
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
| 6 | Quality flags and exclusions of each breath; `is_valid` = no flag and not excluded | `breath_quality`, `exclusions` | notebook 05 §8 |
| 7 | Lung mask from the valid breaths: watershed of the TIV and amplitude maps at 15 % of the maximum TIV, islands < 10 pixels removed | `lung_mask` | eitprocessing `WatershedLungspace`, `FilterROIBySize` |
| 8 | Regions of interest inside the mask and their signals | `regions` | notebook 05 §9 |
| 9 | Auxiliary signals (pressures, flow, volume, device breath markers): "no data" values as NaN, channels without data left out, not filtered, the same units for every vendor | `auxiliary` | - |
| 10 | Airway pressures of each breath: PEEP, plateau, driving pressure | `breath_pressures` | notebook 06 §2, §4 |
| 11 | Inspiratory tidal volume of each breath | `tidal_volume` | - |

Breaths are **flagged, never removed**: the clinician decides what to use.

Quality flags of each breath:

| Column | Rule |
|--------|------|
| `flag_short`, `flag_long` | duration below 0.5 s or above 20 s |
| `flag_suspect_ti_ratio` | Ti/Ttot (from impedance) outside 0.2-0.6 |
| `flag_amplitude_outlier` | amplitude below 0.5× or above 2× the median of the 21 surrounding breaths |
| `flag_duration_outlier` | duration above 2× the median of the 21 surrounding breaths |
| `flag_invalid_data` | within 0.5 s of a gross outlier of the signal |

## 3. Ventilator values of each breath

The auxiliary signals get the same units for every vendor, once, in `auxiliary`:

| Quantity | Unit | From |
|----------|------|------|
| Pressures (airway, oesophageal, PEEP, plateau...) | cmH2O | Dräger mbar × 1.0197 (1 mbar = 100 Pa, 1 cmH2O = 98.0665 Pa); Timpel already cmH2O |
| Flow | L/s | Dräger Medibus L/min ÷ 60; Timpel already L/s |
| Volumes | mL | Timpel L × 1000; Dräger already mL (two Medibus values in L) |
| Derived values | e.g. mL/cmH2O, cmH2O/L/s | Medibus compliance, resistance, elastance |

Minute volumes stay in L/min. `EITRecording.aux_units` gives the unit of each column.

Then each breath gets, in the breath table, the first source available:

| Columns | Source, in order of preference | Channel used |
|---------|-------------------------------|--------------|
| `peep`, `plateau`, `driving_pressure` (cmH2O) | 1. airway pressure waveform: Dräger PressurePod, then Medibus or Timpel `airway_pressure`; 2. without a waveform, the PEEP and plateau computed by the ventilator (Medibus), median over the breath | `pressure_channel` |
| `tidal_volume` (mL, inspiratory) | 1. volume curve: highest volume around the inspiration minus the lowest before it; 2. integral of the inspiratory flow; 3. the tidal volume computed by the ventilator (Medibus: mandatory, then spontaneous inspiratory value, then the generic "tidal volume"), median over the breath | `tidal_volume_channel` |

The pressures are all read on the same channel, so that a constant offset of the
sensor cancels in the driving pressure:

- **start of inspiration**: the last minimum of the pressure before its steepest rise,
  searched near the EIT valley;
- **PEEP**: the median pressure in the 0.2 s before it;
- **plateau**: the median pressure from 0.25 to 0.05 s before the start of
  expiration, only after an end-inspiratory hold of at least 0.5 s. In pressure
  control the flat inspiratory pressure is read the same way;
- **driving pressure** = plateau - PEEP. The peak pressure is never used instead.

The tidal volume is read in a window from half an expiration before the EIT breath
(but not before the middle of the expiration before) to the middle of its own
expiration, so that it holds the whole inspiration even when the EIT valley falls a
little after the start of the inspiratory flow, and nothing of the inspirations around
it. On the flow curve only the run of inspiratory flow around its peak is integrated,
so that noise and cardiac oscillations around zero flow are not counted. In volume
control it is the volume delivered, the one set on the ventilator. If the flow does
not rise during the EIT inspiration of most breaths (inspiration recorded as
negative), the log gives a warning.

A tidal volume typed by the user for the whole recording goes in
`RecordingContext.tidal_volume_ml`: the volume set in volume control, or the mean on
the case report form, only when it is the same for the whole recording (in pressure
control it changes with PEEP: type it per step, [`segmentation.md`](segmentation.md)).
It is kept apart from the measured one: the next layers use the typed one first. A
typed volume outside 100-1500 mL gives a warning (litres typed instead of mL?).

## 4. Output

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
| `breaths` | one row per breath: times and frames, flags, `excluded`, `exclusion_label`, `is_valid`, `peep`, `plateau`, `driving_pressure`, `tidal_volume` |
| `lung_mask`, `tiv_map` | ventilated lung and the mean TIV map it comes from |
| `roi_masks`, `roi_signals` | `ventral`, `dorsal`, `right`, `left`, `layer_1`-`layer_4`, four quadrants |
| `aux`, `aux_units` | auxiliary signals with data, one row per frame (e.g. `airway_pressure_pod`, `esophageal_pressure_pod`, Medibus `peep`, `min_max_flag`), and the unit of each |
| `pressure_channel`, `tidal_volume_channel` | the columns of `aux` the ventilator values of the breaths come from (None if none) |
| `log` | what each step did, with warnings |

Images are seen from the feet, like a CT slice: row 0 is ventral, column 0 is the
patient's right.

Breath frames index every signal of the same length, for example the pressure at
the start and at the peak of a breath:

```python
breath = recording.valid_breaths.iloc[0]
tiv_map = recording.pixels[breath.peak_idx] - recording.pixels[breath.start_idx]
paw = recording.aux.airway_pressure_pod[[breath.start_idx, breath.peak_idx]]
```

## 5. Logs

There are two kinds of record, kept apart on purpose:

- **`EITRecording.log`** belongs to the result: it says which filter ran, what was
  excluded and every warning, and is saved with the data.
- **Technical logs** go through Python `logging` (logger names
  `fasteit.preprocessing.*`). The library never configures logging: the script or
  application decides where the messages go, e.g. `logging.basicConfig(level=logging.INFO)`
  in a notebook, or stdout collected by Docker in an application.

## 6. Limits

- Validated as equivalent to eitprocessing, not against a gold standard: 946 breaths from
  two patients and one volunteer; pressures of each breath on four PEEP trials (notebook 06).
- Breath timing from the impedance: the valley is the minimum of lung volume, not the
  ventilator trigger; asynchrony is not detected; breaths within about 15 s of a sudden
  EELI step are unreliable.
- Without a heart rate the cardiac component is not removed.
- Regions are relative to the ventilated lung: indices of a shift of ventilation use image
  coordinates.
- Verified on Dräger PulmoVista only; adults only (respiratory rate 6-60/min).
- **PEEP** at the airway opening, without intrinsic PEEP (a lower bound; with intrinsic
  PEEP the driving pressure is overestimated). A missing pressure frame near the start of
  a breath loses its PEEP: the breath keeps its EIT values, the PEEP is not estimated.
- **Plateau** only after a hold of at least 0.5 s (or a flat pressure in pressure
  control), not validated against the ventilator plateau; holds, pressure control,
  Medibus and Timpel pressures tested on synthetic data only.
- **Driving pressure** of the respiratory system, valid in passive patients.
- **Tidal volume** from the curves tested on synthetic curves only. It is the volume
  measured by the ventilator: it includes a leak and, if not compensated, the compressible
  volume of the circuit (about 1-2.5 mL per cmH2O); a zero offset of the flow sensor adds
  to it; with a volume curve set back to zero at each breath, the first breaths after a
  decrease of PEEP read too much; the generic Medibus value may be expiratory.

## 7. File layout

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
    ├── auxiliary.py           pressures, flow, volume and device markers, cleaned, common units
    ├── breath_pressures.py    PEEP and plateau of each breath
    ├── tidal_volume.py        inspiratory tidal volume of each breath
    └── pipeline.py            preprocess()
```

*Research use only - not a medical device.*
