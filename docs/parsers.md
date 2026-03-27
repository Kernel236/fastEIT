# fastEIT — Parser Reference

Specifications for all parsers: input formats, output containers, calibration
details, and the pyEIT reconstruction bridge.

For **data container field specifications** see [`data_model.md`](data_model.md).
For **architecture and extension recipes** see [`parsing_layer.md`](parsing_layer.md).

---

## 1. Overview

| File | Vendor | Parser | Container |
|------|--------|--------|-----------|
| `.bin` | Dräger | `DragerBinParser` | `ReconstructedFrameData` |
| `.asc` / `.txt` / `.csv` | Dräger | `DragerAscParser` | `ContinuousSignalData` |
| `.eit` | Dräger | `DragerEitParser` | `RawImpedanceData` |
| `.csv` / `.txt` / `.asc` | Timpel | `TimpelTabularParser` | `ReconstructedFrameData` |
| `.x` | Timpel | `TimpelRawParser` *(scaffold)* | `RawImpedanceData` |

All parsers inherit from `BaseParser` and are accessible via `load_data(path)`.

---

## 2. `DragerBinParser` — `.bin`

**File:** `parsers/draeger/bin/bin_parser.py`

Reads vendor-reconstructed 32×32 pixel frames exported by the PulmoVista 500.
Two frame formats are registered; detection is size-based (no header).

### Frame formats

| Name | Frame size | Medibus channels | Notes |
|------|-----------|-----------------|-------|
| `Draeger_base_4358` | 4358 bytes | 52 | Standard export |
| `Draeger_ext_4382` | 4382 bytes | 58 | PressurePod: adds esophageal + transpulmonary pressure |

### Binary frame layout (base format)

| Field | Type | Size | Content |
|-------|------|------|---------|
| `ts` | float64 | 8 B | Wall-clock timestamp (fraction of day × 86400 → seconds) |
| `dummy` | float32 | 4 B | Unknown — ignored |
| `pixels` | float32 × 1024 | 4096 B | Reconstructed 32×32 image (C-order) |
| `min_max_flag` | int32 | 4 B | +1 = inspiratory peak, −1 = expiratory trough, 0 = none |
| `event_marker` | int32 | 4 B | Event counter |
| `event_text` | S30 | 30 B | ASCII event label, space-padded |
| `timing_error` | int32 | 4 B | 0 = frame timing OK |
| `medibus_data` | float32 × 52 | 208 B | Ventilator Medibus channels; sentinel −3.4×10³⁸ = disconnected |

PressurePod format (`Draeger_ext_4382`) adds 6 extra Medibus channels (indices 52–57):
esophageal pressure (Pes), airway pressure (Paw), transpulmonary pressure (Ptp),
and three auxiliary pressure waveforms from the Pod hardware.

### Sentinel handling

Medibus channels disconnected from the ventilator carry the sentinel value
`−3.4×10³⁸` (IEEE 754 bit pattern `0xFF7FC99E`). The parser replaces these
with `NaN` so downstream code can use `np.isnan()` to detect disconnected channels.

### Sampling frequency

Not stored in the binary. Estimated from consecutive timestamp differences:
`fs = 1 / median(diff(ts))`, rounded to the nearest integer Hz.

---

## 3. `DragerEitParser` — `.eit`

**File:** `parsers/draeger/eit/eit_parser.py`

Reads raw transimpedance measurements from the PulmoVista 500 proprietary
binary format. This is the **only open-source Python parser for this format**.

### File structure

```
Bytes 0–11          : preamble — 3 × int32 LE
                        [0:4]  format_version (always 51)
                        [4:8]  sep_offset      (file-specific, variable)
                        [8:12] unknown_int
Bytes 12–sep_offset : ASCII header ("key: value\r\n" lines, latin-1)
Bytes sep_offset–+7 : separator b'**\r\n\r\n\r\n' (8 bytes)
Bytes sep_offset+8… : binary frames, 5495 bytes each
```

### ASCII header fields

| Header key | Metadata key | Type | Notes |
|-----------|-------------|------|-------|
| `Framerate [Hz]` | `fs` | float | Acquisition rate (typically 50.0 Hz) |
| `Date` | `date` | str | DD.MM.YYYY |
| `Time` | `time` | str | HH:MM:SS.mmm |
| `Frequency [kHz]` | `frequency_khz` | float | Excitation frequency (≈101.5 kHz) |
| `Amplitude [uA]` | `amplitude_ua` | float | Injection current amplitude (≈9100 µA) |
| `Gain` | `gain` | int | Hardware amplifier gain |
| `Samples` | `samples_per_period` | int | ADC samples per excitation period |
| `Periods` | `periods` | int | Excitation periods per measurement |

### Binary frame layout (5495 bytes)

| Field | Type | Size | Content |
|-------|------|------|---------|
| `timestamp` | float64 | 8 B | Fraction of day |
| `unknown_f8` | float64 | 8 B | Unknown |
| `trans_A` | float64 × 208 | 1664 B | Primary transimpedance set (ADC counts) |
| `unknown_16a` | float64 × 16 | 128 B | Unknown |
| `injection_current` | float64 × 16 | 128 B | Injected current per drive pattern (ADC counts) |
| `unknown_16b` | float64 × 16 | 128 B | Unknown |
| `voltage_A` | float64 × 16 | 128 B | Electrode voltage A (ADC counts) |
| `unknown_50` | float64 × 50 | 400 B | Unknown |
| `trans_B` | float64 × 208 | 1664 B | Reference transimpedance set (ADC counts) |
| `unknown_48` | float64 × 48 | 384 B | Unknown |
| `voltage_B` | float64 × 16 | 128 B | Electrode voltage B (ADC counts) |
| `unknown_6` | float64 × 6 | 48 B | Unknown |
| `gugus` | float64 × 44 | 352 B | Ignored (EIDORS terminology) |
| `unknown_byte` | uint8 | 1 B | Unknown |
| `medibus` | float32 × 67 | 268 B | Ventilator Medibus channels (67 in `.eit`, 52 in `.bin`) |
| `event_text` | S30 | 30 B | ASCII event label |
| `mixed` | uint8 × 24 | 24 B | Mixed content |
| `frame_counter` | uint16 | 2 B | Rolling frame counter |
| `padding` | uint8 × 2 | 2 B | Alignment padding |

### Calibration pipeline

Empirical constants from EIDORS `read_draeger_file.m` (A. Adler, estimated 2016-04-07).
These are not present in the `.eit` header — they are hardware-specific and hardcoded.

| Constant | Value | Role |
|----------|-------|------|
| `FT_A` | 0.00098242 | Transimpedance scaling — primary channel |
| `FT_B` | 0.00019607 | Transimpedance scaling — reference channel |
| `FC_CURRENT` | 194326.3536 | ADC counts → Ampere |
| `FV_VOLTAGE` | 0.11771 | ADC counts → Volt |

```python
vv      = FT_A * trans_A - FT_B * trans_B   # (N_frames, 208) calibrated transimpedance [Ω]
I_real  = injection_current / FC_CURRENT     # (N_frames, 16)  injected current [A]
V_diff  = (voltage_A - voltage_B) / FV_VOLTAGE  # (N_frames, 16) differential voltage [V]
```

### Measurement protocol

208 measurements per frame = 16 electrodes × 13 measurement pairs.
Adjacent current injection (drive distance = 1 electrode), adjacent voltage
measurement (step = 1 electrode), 3 pairs excluded per drive pattern
(those sharing a voltage electrode with the current-injection pair).

The ordering of the 208 values within each frame follows the **`fmmu`** (rotating)
protocol: for each drive pattern the voltage sweep starts from the electrode
immediately adjacent to the current sink and rotates around the ring.
Confirmed from the PulmoVista 500 IFU SW 1.3n p. 144 measurement diagram.

---

## 4. `DragerAscParser` — `.asc`

**File:** `parsers/draeger/asc/asc_parser.py`

Reads the PulmoVista 500 continuous waveform export (tab-separated text).
Returns `ContinuousSignalData` with one row per EIT frame.

Detection: looks for `"Dräger"` or `"Draeger"` keyword in the first 40 lines.
Sampling frequency estimated from the `time` column if present.

Typical columns: `image`, `time`, `global`, `local_1_x_16_y_29` … `local_4_x_16_y_05`,
`minmax`, `event`, `eventtext`, `timing_error`, plus Medibus channel columns when
the ventilator is connected via Medibus.

---

## 5. `TimpelTabularParser` — `.csv` / `.txt` / `.asc`

**File:** `parsers/timpel/timpel_parser.py`

Reads Timpel EIT device tabular exports. Each row is one EIT frame;
columns 0–1023 are the 32×32 pixel values (C-order); columns 1024–1029
are auxiliary signals (airway_pressure, flow, volume, min_flag, max_flag, qrs_flag).

Returns `ReconstructedFrameData`. Sampling frequency defaults to 50 Hz if a
`first_frame` column is present but timing cannot be derived otherwise.

Detection: looks for `"Timpel"` keyword in the first 40 lines.

---

## 6. GREIT reconstruction bridge

**File:** `parsers/draeger/eit/eit_pyeit_bridge.py`
**Optional dependency:** `pip install fasteit[pyeit]`

Converts `RawImpedanceData.measurements` (N_frames, 208) to reconstructed
32×32 images (N_frames, 32, 32) using the GREIT algorithm.

Reference: Adler A et al., "GREIT: a unified approach to 2D linear EIT
reconstruction of lung images." *Physiol. Meas.* 30 (2009) S35–S55.
DOI: [10.1088/0967-3334/30/6/S03](https://doi.org/10.1088/0967-3334/30/6/S03)

### `build_greit(n_el, h0, p, lamb, n) → (solver, protocol)`

Builds a pyEIT GREIT solver on a circular unit-disk mesh.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_el` | 16 | Number of electrodes |
| `h0` | 0.1 | Mesh density |
| `p` | 0.2 | GREIT noise figure (Adler 2009 recommendation) |
| `lamb` | 1e-2 | Tikhonov regularisation |
| `n` | 32 | Output image side length in pixels |

### `reconstruct_greit(measurements, ref_frame, …) → ndarray`

| `ref_frame` value | Behaviour |
|-------------------|-----------|
| `None` | Mean of all frames in this recording (default) |
| `int` | Single frame index |
| `(start, end)` | Mean of `measurements[start:end]` |
| `np.ndarray (208,)` | External reference — for cross-recording EELI comparison |

**Sign convention**: GREIT outputs conductivity change (Δσ); the function negates
it to match the Dräger impedance convention (air → impedance rises → positive peak).

**Image orientation** — three steps applied inside `reconstruct_greit()`, all
sourced from PulmoVista 500 IFU SW 1.3n:
1. **Negate** (sign flip): impedance convention — see above.
2. **`rot90(k=1)`** (90° CCW): electrode 1 sits at 12 o'clock on the PulmoVista
   belt (anterior/ventral wall); pyEIT places electrode 0 at 3 o'clock. Rotation
   moves the anterior region to the top of the image. Source: IFU p. 144.
3. **`fliplr`** (horizontal mirror): Dräger displays images in caudal-cranial
   projection (CT convention viewed from feet): left side of image = right side
   of patient. pyEIT produces the anatomical mirror; `fliplr` corrects this.
   Source: IFU p. 146.

**Spatial quality note**: a generic circular mesh does not replicate Dräger's
proprietary reconstruction. Both algorithms produce correct relative distributions,
but the vendor `.bin` uses a device-specific mesh and matrix. A data-driven
reconstruction that learns the Dräger mapping from paired `.eit`/`.bin` recordings
is provided in `fasteit.reconstruction` (see `feat/ml-reconstruction`).

---

## 7. Loader utilities

**File:** `parsers/loader.py`

### `load_data(path) → BaseData`

Single entry point. Auto-detects vendor and format, selects the right parser,
returns the appropriate container. Raises `ValueError` on unknown format.

```python
from fasteit.parsers.loader import load_data

data = load_data("patient01.bin")   # → ReconstructedFrameData
data = load_data("patient01.eit")   # → RawImpedanceData
data = load_data("patient01.asc")   # → ContinuousSignalData
```

### `load_many(paths) → list[BaseData]`

Parses an explicit list of paths in order. Formats can be mixed.

```python
from fasteit.parsers.loader import load_many

recordings = load_many(["patient01.bin", "patient02.bin", "patient01.asc"])
```

### `load_folder(folder, pattern="*") → list[BaseData]`

Scans a directory, parses every file with a registered parser.
Files that raise any error are silently skipped (warning to stderr).

```python
from fasteit.parsers.loader import load_folder

all_bin  = load_folder("/data/session01/", pattern="*.bin")
all_data = load_folder("/data/session01/")          # mixed formats
recursive = load_folder("/data/", pattern="**/*.bin")
```
