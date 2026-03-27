# fastEIT — Data Model

Specifications for all data containers. For parsing flow, detection logic, and
extension recipes see [`parsing_layer.md`](parsing_layer.md).

---

## 1. Overview

| File | Vendor | Container | Content |
|------|--------|-----------|---------|
| `.bin` | Dräger | `ReconstructedFrameData` | 32×32 pixel matrices + Medibus signals |
| `.csv` / `.txt` / `.asc` | Timpel | `ReconstructedFrameData` | 32×32 pixel matrices + 6 device signals |
| `.asc` | Dräger | `ContinuousSignalData` | Frame-by-frame signal table, no matrices |
| `.eit` | Dräger | `RawImpedanceData` | 208 raw transimpedances per frame |
| `.x`   | Timpel | `RawImpedanceData` | 208 raw transimpedances per frame |

All containers inherit from `BaseData` (`models/base_data.py`).

---

## 2. `BaseData` — common base

```
filename    str          Source file path
file_format str          "bin" | "eit" | "asc" | "csv" | "txt"
vendor      str          "draeger" | "timpel"
fs          float|None   Sampling frequency in Hz (parser-derived)
metadata    dict         Header fields, firmware version, warnings, etc.
n_frames    int          Set by subclass __post_init__ — never passed by caller
duration    float        n_frames / fs  (0.0 if fs is None)
```

---

## 3. `ReconstructedFrameData`

**File:** `models/reconstructed_data.py`
**Produced by:** `DragerBinParser` (`.bin`), `TimpelTabularParser` (`.csv/.txt/.asc`)

```
frames      np.ndarray   Structured array shape (N,). Named fields:
                           ts            float64   timestamp (seconds)
                           pixels        float32   32×32 image per frame
                         Dräger .bin only:
                           min_max_flag  int32     +1=insp peak, -1=exp trough, 0=none
                           event_marker  int32     event counter
                           event_text    S30       null-padded ASCII event string
                           timing_error  int32     0 = ok

aux_signals dict|None    Named arrays, each shape (N,), frame-aligned with frames.
                         Dräger .bin: 52 or 58 Medibus channels.
                         Timpel: 6 device channels (airway_pressure, flow,
                         volume, min_flag, max_flag, qrs_flag — cols 1024-1029).
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

**Timestamp conventions:**

| Vendor | `ts` field | Unit |
|--------|-----------|------|
| Dräger `.bin` | wall-clock fraction-of-day from device | seconds (converted from float64 day fraction) |
| Timpel `.csv` | synthetic: `(frame_index + first_frame) / 50.0` | seconds from start |

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
**Produced by:** `DragerEitParser` (`.eit`); `TimpelRawParser` (`.x`) — scaffold

```
measurements   np.ndarray   shape (N_frames, 208)
                            Calibrated transimpedances (not raw ADC counts, not
                            yet image-reconstructed). For Dräger:
                            vv = FT_A*trans_A - FT_B*trans_B (EIDORS, Adler 2016)
                            208 = 16 electrodes × 13 measurement pairs
                            (adjacent drive, standard Dräger pattern)

aux_signals    dict|None    Named per-frame arrays. Populated by DragerEitParser:
                              "timestamp"          float64 (N,)    fraction of day
                              "trans_A"            float64 (N,208) raw ADC — primary
                              "trans_B"            float64 (N,208) raw ADC — reference
                              "injection_current"  float64 (N,16)  raw ADC counts
                              "I_real"             float64 (N,16)  injected current [A]
                              "voltage_A"          float64 (N,16)  raw ADC counts
                              "voltage_B"          float64 (N,16)  raw ADC counts
                              "V_diff"             float64 (N,16)  differential voltage [V]
                              "frame_counter"      uint16  (N,)
                              "medibus"            float32 (N,67)  ventilator channels
                            None if the source file carries no auxiliary signals.
```

Intended consumer: **pyEIT** or any reconstruction library.
fastEIT does not perform image reconstruction — it ingests vendor-reconstructed files.
