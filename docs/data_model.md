# fastEIT — Data Model

Specifications for all data containers. For parsing flow, detection logic, and
extension recipes see [`parsing_layer.md`](parsing_layer.md).

---

## 1. Overview

| File | Vendor | Container | Content |
|------|--------|-----------|---------|
| `.bin` | Dräger | `ReconstructedFrameData` | 32×32 pixel matrices + Medibus signals |
| `.txt` | Timpel | `ReconstructedFrameData` | 32×32 pixel matrices + 5 device signals |
| `.asc` | Dräger | `ContinuousSignalData` | Frame-by-frame signal table, no matrices |
| `.eit` | Dräger | `RawImpedanceData` | 208 raw transimpedances per frame |
| `.x`   | Timpel | `RawImpedanceData` | 208 raw transimpedances per frame |

All containers inherit from `BaseData` (`models/base_data.py`).

---

## 2. `BaseData` — common base

```
filename    str          Source file path
file_format str          "bin" | "eit" | "asc" | "txt" | "x"
vendor      str          "draeger" | "timpel"
fs          float|None   Sampling frequency in Hz (parser-derived)
metadata    dict         Header fields, firmware version, warnings, etc.
n_frames    int          Set by subclass __post_init__ — never passed by caller
duration    float        n_frames / fs  (0.0 if fs is None)
```

---

## 3. `ReconstructedFrameData`

**File:** `models/reconstructed_data.py`
**Produced by:** `DragerBinParser` (`.bin`), `TimpelTabularParser` (`.txt`)

```
frames      np.ndarray   Structured array shape (N,). Named fields:
                           ts            float64   timestamp (fraction of day)
                           pixels        float32   32×32 image per frame
                         Dräger .bin only:
                           min_max_flag  int32     +1=insp peak, -1=exp trough, 0=none
                           event_marker  int32     event counter
                           event_text    S30       null-padded ASCII event string
                           timing_error  int32     0 = ok

aux_signals dict|None    Named arrays, each shape (N,), frame-aligned with frames.
                         Dräger .bin: 52 or 58 Medibus channels.
                         Timpel .txt: 5 device channels.
                         None if the source file carries no auxiliary signals.
```

**Property accessors:**

```python
data.timestamps      # frames["ts"]           shape (N,)
data.pixels          # frames["pixels"]        shape (N, 32, 32)
data.min_max_flags   # frames["min_max_flag"]  shape (N,)  — Dräger .bin only
data.event_markers   # frames["event_marker"]  shape (N,)  — Dräger .bin only
data.event_texts     # frames["event_text"]    shape (N,)  — Dräger .bin only
data.global_signal   # pixels.sum(axis=(1,2))  shape (N,)
data.roi_signals     # 4 horizontal strips     shape (N, 4)
data.roi_signal(n)   # single ROI 0–3          shape (N,)
```

`global_signal` and `roi_signals` are raw sums with no lung mask — the mask
is applied in the preprocessing layer.

---

## 4. `ContinuousSignalData`

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

## 5. `RawImpedanceData`

**File:** `models/raw_impedance_data.py`
**Produced by:** `DragerEitParser` (`.eit`), `TimpelRawParser` (`.x`) — both scaffold

```
measurements   np.ndarray   shape (N_frames, 208)
                            208 = 16 electrodes × 13 measurement pairs
                            Adjacent drive pattern (Dräger and Timpel)
```

Intended consumer: **pyEIT** or any reconstruction library.
fastEIT does not perform image reconstruction — it ingests vendor-reconstructed files.

---

## 6. File layout

```
src/fasteit/
├── models/
│   ├── base_data.py           BaseData
│   ├── reconstructed_data.py  ReconstructedFrameData
│   ├── continuous_data.py     ContinuousSignalData
│   └── raw_impedance_data.py  RawImpedanceData
│
└── parsers/
    ├── base.py                BaseParser ABC
    ├── errors.py              Exception hierarchy
    ├── bin_formats.py         FormatSpec registry (BIN_FORMAT_SPECS)
    ├── detection.py           Auto-detection: extension, vendor, format
    ├── loader.py              load_data() + parser registry
    ├── draeger/
    │   ├── bin/
    │   │   ├── draeger_dtypes.py  FRAME_BASE/EXT_DTYPE, MEDIBUS field lists
    │   │   ├── bin_parser.py      DragerBinParser
    │   │   └── bin_utils.py       sentinel detection, fs estimation
    │   ├── asc/               DragerAscParser
    │   └── eit/               DragerEitParser (scaffold)
    └── timpel/
        ├── timpel_dtypes.py   Timpel field definitions (scaffold)
        └── timpel_parser.py   TimpelTabularParser (scaffold)
```
