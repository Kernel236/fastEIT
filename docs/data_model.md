# fastEIT - Data Model

Specifications for all data containers. For parser details, calibration
constants, and loader utilities see [`parsers.md`](parsers.md). For architecture
and extension recipes see [`parsing_layer.md`](parsing_layer.md).

All containers of layer 1 inherit from `BaseData` (`models/base_data.py`). The
objects of the later layers are in [section 5](#5-from-a-file-to-the-clinical-indices).

---

## 1. `BaseData` - common base

```
filename    str          Source file path
file_format str          "bin" | "eit" | "asc" | "csv" | "txt"
vendor      str          "draeger" | "timpel"
fs          float|None   Sampling frequency in Hz (parser-derived)
metadata    dict         Header fields, firmware version, warnings, etc.
n_frames    int          Set by subclass __post_init__ - never passed by caller
duration    float        n_frames / fs  (0.0 if fs is None)
```

---

## 2. `ReconstructedFrameData`

**File:** `models/reconstructed_data.py`
**Produced by:** `DragerBinParser` (`.bin`), `TimpelTabularParser` (`.csv/.txt/.asc`)

```
frames      np.ndarray   Structured array shape (N,). Named fields:
                           ts            float64   timestamp (unit depends on vendor, see below)
                           pixels        float32   32×32 image per frame
                         Dräger .bin only:
                           min_max_flag  int32     +1=insp peak, -1=exp trough, 0=none
                           event_marker  int32     event counter
                           event_text    S30       ASCII event string, space- or null-padded
                           timing_error  int32     0 = ok

aux_signals dict|None    Named arrays, each shape (N,), frame-aligned with frames.
                         Dräger .bin: 52 or 58 Medibus channels, sentinel values
                         left unchanged (replaced in preprocessing).
                         Units of the device (parser tables MEDIBUS_UNITS,
                         TIMPEL_AUX_UNITS); preprocessing converts them to the
                         same units for every vendor (cmH2O, L/s, mL).
                         Timpel: 6 device channels (airway_pressure, flow,
                         volume, min_flag, max_flag, qrs_flag - cols 1024-1029).
                         None if the source file carries no auxiliary signals.
```

**Property accessors:**

```python
data.timestamps      # frames["ts"]           shape (N,)
data.pixels          # frames["pixels"]        shape (N, 32, 32)
data.min_max_flags   # frames["min_max_flag"]  shape (N,)  - Dräger .bin only
data.event_markers   # frames["event_marker"]  shape (N,)  - Dräger .bin only
data.event_texts     # frames["event_text"]    shape (N,)  - Dräger .bin only
data.global_signal   # pixels.sum(axis=(1,2))  shape (N,)
data.roi_signals     # 4 horizontal strips     shape (N, 4)
data.roi_signal(n)   # single ROI 0-3          shape (N,)
```

`global_signal` and `roi_signals` are raw sums with no lung mask - the mask
is applied in the preprocessing layer. They sum the values as stored, Dräger "no data"
values included. On the parsed file, `fasteit.preprocessing.pipeline` has:

- `global_impedance(raw)` - the sum of the pixels with a value, NaN where a frame has
  missing pixels (the frames `preprocess()` excludes), read in blocks for long files;
- `pixel_values(raw, index)` - the pixels of some frames, "no data" values as NaN;
- `recording_time(raw)` - the time of each frame from the device clock. `duration` is
  `n_frames / fs`: when the device dropped frames, the file lasts a little longer.

**Timestamp conventions:**

| Vendor | `ts` field | Unit |
|--------|-----------|------|
| Dräger `.bin` | wall-clock time from device, stored as-is | fraction of day (× 86400 for seconds) |
| Timpel `.csv` | synthetic: `(frame_index + first_frame) / 50.0` | seconds from start |

---

## 3. `ContinuousSignalData`

**File:** `models/continuous_data.py`
**Produced by:** `DragerAscParser` (`.asc`)

```
table   pd.DataFrame   One row per EIT frame. Snake_case column names.
                       Dräger .asc columns include:
                         image, time, global, local_1_x_16_y_29 ... local_4_x_16_y_05
                         minmax, event, eventtext, timing_error
                         + Medibus channels when ventilator is connected via Medibus
```

`n_frames` = `len(table)`. `duration` = `n_frames / fs`.

---

## 4. `RawImpedanceData`

**File:** `models/raw_impedance_data.py`
**Produced by:** `DragerEitParser` (`.eit`)

```
measurements   np.ndarray   shape (N_frames, 208)
                            Calibrated transimpedances (not raw ADC counts, not
                            yet image-reconstructed). For Dräger:
                            vv = FT_A*trans_A - FT_B*trans_B (EIDORS, Adler 2016)
                            208 = 16 electrodes × 13 measurement pairs
                            (adjacent drive, standard Dräger pattern)

aux_signals    dict|None    Named per-frame arrays. Populated by DragerEitParser:
                              "timestamp"          float64 (N,)    fraction of day
                              "trans_A"            float64 (N,208) raw ADC - primary
                              "trans_B"            float64 (N,208) raw ADC - reference
                              "injection_current"  float64 (N,16)  raw ADC counts
                              "I_real"             float64 (N,16)  injected current [A]
                              "voltage_A"          float64 (N,16)  raw ADC counts
                              "voltage_B"          float64 (N,16)  raw ADC counts
                              "V_diff"             float64 (N,16)  differential voltage [V]
                              "frame_counter"      uint16  (N,)
                              "medibus"            float32 (N,67)  ventilator channels
                            None if the source file carries no auxiliary signals.
```

Intended consumer: `reconstruct_greit()` (in `fasteit.parsers.draeger.eit.eit_pyeit_bridge`,
requires optional `pyeit` dependency) or any external reconstruction library.
For `.bin` and Timpel `.csv` files, fastEIT ingests vendor-reconstructed images
and does not re-reconstruct them.

---

## 5. From a file to the clinical indices

Each layer returns one object that holds the object of the layer before it, so the last
one holds everything:

```
Dräger .bin, Timpel .csv / .txt / .asc
 └─ load_data() ───────────► ReconstructedFrameData   layer 1: the file as it is (32x32 images)
     └─ preprocess() ──────► EITRecording             layer 2: the measured breaths
          ├─ tidal_features() ───────► TidalFeatures       layer 3: indices
          └─ segment_peep_trial() ───► PeepTrial           layer 2b: PEEP steps
               └─ peep_trial_features() ► PeepTrialFeatures  layer 3: indices
                    └─ recording_tables() ► tables, CSV        layer 4: export
```

The pipeline starts from reconstructed images. A Dräger `.eit` file holds the raw
measurements (`RawImpedanceData`, 208 transimpedances per frame) and needs a
reconstruction first; a Dräger `.asc` export holds signals only (`ContinuousSignalData`)
and has no images ([parsers](parsers.md)).

Each value is stored once, in the table of the layer that produces it; tables are joined
on `breath_id`, `step` (with `peep`) and `region`.

| Where | One row per | Content |
|-------|-------------|---------|
| `EITRecording.breaths` | breath | times and frames, quality flags, `excluded`, `is_valid`, PEEP, plateau, driving pressure, measured tidal volume |
| `EITRecording.aux` (+ `aux_units`) | frame | auxiliary signals in cmH2O, L/s, mL; device markers |
| `EITRecording` arrays | - | `pixels`, `lung_mask`, `roi_masks`, `roi_signals`, `global_signal`, `time` |
| `EITRecording.context` | - | `RecordingContext`: `patient_id`, `label`, `position`, `protocol`, `interface`, `tidal_volume_ml`, `notes` |
| `PeepTrial.breaths` | breath | the breaths of the recording plus `peep_step`, `analysed` |
| `PeepTrial` fields | - | `source` (`"pressure"` or `"manual"`), `phase` (the PEEP trial part of the recording, or None), `config` (`SegmentationConfig`) |
| `PeepTrial.steps` | step | PEEP level measured and of the protocol (`peep_set`), times, breaths analysed, `enough`, plateau, driving pressure and tidal volume with their sources |
| `TidalFeatures.breaths`, `PeepTrialFeatures.breaths` | breath | EELI, TIV, GI, centre of ventilation, respiratory rate, tidal volume used, compliance (EIT units and mL/cmH2O) |
| `...Features.regions` | breath and region | EELI, TIV, share, TIV in mL, compliance (EIT units and mL/cmH2O) |
| `PeepTrialFeatures.steps` | step | medians of the analysed breaths, compliance and % of the best, collapse, overdistension, changes vs the previous and the first step |
| `PeepTrialFeatures.step_regions` | step and region | the regional values of each step |
| `TidalFeatures.summary` | region and index | median and quartiles over the valid breaths |
| `PeepTrialFeatures.summary` | region and feature | crossing PEEP, best PEEP, R/I steps (global), R/I and recruited volume (global and by region) |
| `PeepTrialFeatures.config` | - | `AnalysisConfig` used (R/I levels) |

Layer 4 joins them into flat tables, one row per breath or step with everything, keyed
by patient, recording and phase ([`export.md`](export.md)). Details: [`preprocessing.md`](preprocessing.md),
[`segmentation.md`](segmentation.md), [`features.md`](features.md).

