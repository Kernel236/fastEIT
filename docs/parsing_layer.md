# fastEIT — Parsing Layer: How It Works and How to Extend It

This document explains the architecture of the parsing layer and provides
step-by-step recipes for the most common extension tasks:

1. [How the parsing layer works](#1-how-the-parsing-layer-works)
2. [Add a new Dräger `.bin` frame size](#2-add-a-new-drger-bin-frame-size)
3. [Add a new parser](#3-add-a-new-parser)
4. [Testing checklist](#4-testing-checklist)
5. [File layout](#5-file-layout)

---

## 1. How the parsing layer works

### The single entry point

Everything starts from `load_data(path)` in `parsers/loader.py`. You call it
with a file path; it returns the right data container without you knowing which
parser was used.

```python
from fasteit.parsers.loader import load_data

data = load_data("patient01.bin")   # → ReconstructedFrameData
data = load_data("patient01.asc")   # → ContinuousSignalData
```

Internally it does three things in sequence:

```
1. detect_vendor_and_format(path)   ← What file is this?
2. build_parser_from_detection(...)  ← Which parser handles it?
3. parser.parse_safe(path)           ← Read and return data
```

### Detection (`parsers/detection.py`)

Detection maps a file to `(vendor, extension, optional_bin_format)`.
The strategy depends on the extension:

| Extension | Strategy | How it works |
|-----------|----------|--------------|
| `.bin` | size-based | `file_size % frame_size == 0` for each registered `FormatSpec` |
| `.eit` | header-based | reads first bytes, looks for ASCII magic string for each registered `HeaderFormatSpec` |
| `.asc` `.txt` `.csv` | content-based | reads first 40 lines, looks for vendor keyword |

`.bin` is special: there is no header. The only way to identify the format is
that the file size must be an exact multiple of the frame size in bytes. The
`BIN_FORMAT_SPECS` list in `parsers/bin_formats.py` holds every known frame
size. If only one spec matches → that's the format. If zero match → error.
If more than one match → `AmbiguousFormatError` (rare, requires two frame
sizes that divide the same file size).

### Parser registry (`parsers/loader.py`)

`default_parser_registry()` returns a plain dict:

```python
{
    ("draeger", ".bin"): lambda: DragerBinParser(),
    ("draeger", ".eit"): lambda: DragerEitParser(),
    ("draeger", ".asc"): lambda: DragerAscParser(),
    ("draeger", ".txt"): lambda: DragerAscParser(),
    ("draeger", ".csv"): lambda: DragerAscParser(),
    ("timpel",  ".csv"): lambda: TimpelTabularParser(),
    ("timpel",  ".txt"): lambda: TimpelTabularParser(),
    ("timpel",  ".asc"): lambda: TimpelTabularParser(),
}
```

Key = `(vendor_lowercase, extension_with_dot_lowercase)`.
Value = zero-argument factory that returns a fresh parser instance.

### `BaseParser` and `parse_safe`

Every parser inherits from `BaseParser` (Template Method pattern):

```
BaseParser
  ├── validate(path) → bool        ← you implement this
  ├── parse(path) → BaseData       ← you implement this
  └── parse_safe(path) → BaseData  ← free: validates then parses
```

`parse_safe` raises `FileNotFoundError` if the file is missing and
`ValueError` if `validate()` returns False. Callers always use `parse_safe`
(or `load_data`, which calls it). Direct `parse()` skips validation and is
used internally or in tests.

### Data flow summary

```
load_data("patient01.bin")
    │
    ├─ detect_vendor_and_format()
    │   └─ file_size 4358000 % 4358 == 0
    │       → FormatSpec("Draeger_base_4358", vendor="draeger", frame_size=4358)
    │       → FileDetection(extension=".bin", vendor="draeger", bin_format=spec)
    │
    ├─ build_parser_from_detection()
    │   └─ registry[("draeger", ".bin")] → DragerBinParser()
    │
    ├─ parser.parse_safe("patient01.bin")
    │   ├─ validate() → file exists, size divisible by known frame size → True
    │   └─ parse()
    │       ├─ np.memmap with spec.dtype
    │       ├─ fs estimation from timestamps
    │       └─ ReconstructedFrameData(frames, aux_signals, fs, ...)
    │
    └─ attach detected vendor/extension to data.metadata
       return ReconstructedFrameData
```

### Multi-file loaders

Two convenience wrappers built on top of `load_data()`:

**`load_many(paths)`** — takes an explicit list of paths, returns a list of
`BaseData` in the same order. Vendor and format can be mixed freely.

```python
from fasteit.parsers.loader import load_many

recordings = load_many([
    "patient01.bin",
    "patient02.bin",
    "patient01.asc",
])
```

**`load_folder(folder, pattern="*")`** — scans a directory (alphabetically),
parses every file that has a registered parser, and **silently skips** files
that raise any error (no registered parser, malformed file, wrong extension).
Skipped files print a one-line warning to `stderr` but never abort the scan.

```python
from fasteit.parsers.loader import load_folder

# All files in a folder (mixed formats)
recordings = load_folder("/data/session01/")

# Only .bin files
recordings = load_folder("/data/session01/", pattern="*.bin")

# Recursive search
recordings = load_folder("/data/", pattern="**/*.bin")
```

Both functions accept the same optional `registry=` argument as `load_data()`,
making it easy to inject a custom parser registry for testing or for formats
not yet registered by default.

---

## 2. Add a new Dräger `.bin` frame size

If a new PulmoVista firmware introduces a new frame layout (e.g., 4406
bytes with 6 additional Medibus fields).

**Files to change: 2. Everything else is automatic.**

### Step 1 — Define the dtype

File: `src/fasteit/parsers/draeger/bin/draeger_dtypes.py`

Add the new structured dtype and field list following the existing pattern.
The layout must sum exactly to the new frame size — verify with a comment.

```python
# New extended frame (4406 bytes) — hypothetical example
FRAME_NEWEXT_DTYPE = np.dtype([
    ("ts",           "<f8"),          #   8 bytes
    ("dummy",        "<f4"),          #   4 bytes
    ("pixels",       "<f4", (32, 32)),# 4096 bytes
    ("min_max_flag", "<i4"),          #   4 bytes
    ("event_marker", "<i4"),          #   4 bytes
    ("event_text",   "S30"),          #  30 bytes
    ("timing_error", "<i4"),          #   4 bytes
    ("medibus_data", "<f4", (64,)),   # 256 bytes  ← was 208 or 232
])
# Total: 8+4+4096+4+4+30+4+256 = 4406 bytes

MEDIBUS_NEWEXT_FIELDS: list[tuple[str, str, bool]] = _MEDIBUS_COMMON + [
    # idx 51: first new field
    ("high_pressure", "mbar", False),
    # idx 52-63: remaining new fields ...
]

MEDIBUS_NEWEXT_INDEX: dict[str, int] = {
    name: i for i, (name, _, _) in enumerate(MEDIBUS_NEWEXT_FIELDS)
}
```

**Rule:** the frame layout must be byte-exact. Add the arithmetic comment
(`# Total: ...`) and verify it matches the new frame size before continuing.

### Step 2 — Register the format spec

File: `src/fasteit/parsers/bin_formats.py`

```python
from fasteit.parsers.draeger.bin.draeger_dtypes import (
    FRAME_NEWEXT_DTYPE,
    MEDIBUS_NEWEXT_FIELDS,
)

BIN_FORMAT_SPECS: tuple[FormatSpec, ...] = (
    FormatSpec(
        name="Draeger_base_4358",
        ...
    ),
    FormatSpec(
        name="Draeger_ext_4382",
        ...
    ),
    FormatSpec(                                      # ← add this
        name="Draeger_newext_4406",
        vendor="draeger",
        frame_size_bytes=4406,
        dtype=FRAME_NEWEXT_DTYPE,
        medibus_fields=tuple(name for name, _, _ in MEDIBUS_NEWEXT_FIELDS),
        has_pressure_pod_fields=True,
    ),
)
```

**That's it.** `detect_bin_format_from_size()` picks it up automatically.
`DragerBinParser.parse()` uses `spec.dtype` and `spec.medibus_fields` from
the spec — no parser changes needed.

---

## 3. Add a new parser

This recipe covers any combination: new vendor or existing vendor, binary or
tabular. The five steps are always the same; only the content of steps 1 and 2
differs by format type.

**Return type by format:**

| Format | Return type | Example |
|--------|-------------|---------|
| Binary frame sequence | `ReconstructedFrameData` | Dräger `.bin`, AcmEIT `.acm` |
| Tabular (CSV / TXT) | `ContinuousSignalData` | Dräger `.asc`, Hamilton export, SERVO-U export, capnograph |

Tabular parsers are common for any device that exports to a plain-text table —
including standalone ventilators (Hamilton, Maquet SERVO, Dräger Evita) that
export continuous waveforms (Flow, Paw, Volume) and/or breath-averaged
parameters (TV, RR, PEEP, compliance) in the same file.

**Files to change: 4–5.**

---

### Step 1 — Detection

Detection works the same way for all three file families: **register a spec, the
generic function finds it automatically**. You rarely need to touch `detection.py`.

---

**`.eit`-style (header + binary frames, same `.eit` extension, new vendor)**

File: `src/fasteit/parsers/header_formats.py`

Just add a `HeaderFormatSpec` and append it to `HEADER_FORMAT_SPECS`.
`detect_vendor_from_eit_header()` iterates the list and returns the first match —
no changes to `detection.py` needed.

```python
CAREFUSION_EIT_SPEC = HeaderFormatSpec(
    name="Carefusion_EIT_v1",
    vendor="carefusion",
    magic_string="CareFusion",      # ASCII substring in first magic_search_bytes bytes
    magic_search_bytes=256,
    encoding="utf-8",
    frame_size_bytes=5120,          # byte-exact frame size for this firmware
    n_electrodes=16,
    n_measurements=208,
)

HEADER_FORMAT_SPECS: tuple[HeaderFormatSpec, ...] = (
    DRAEGER_EIT_HEADER_SPEC,
    CAREFUSION_EIT_SPEC,            # ← add here; detection is automatic
)
```

`HeaderFormatSpec` fields:

| Field | Purpose |
|-------|---------|
| `magic_string` | ASCII substring searched in the first `magic_search_bytes` |
| `magic_search_bytes` | How many bytes to read for detection (keep it small) |
| `encoding` | Header text encoding (usually `"latin-1"` or `"utf-8"`) |
| `frame_size_bytes` | Byte-exact size of one binary frame — used by the parser |
| `n_electrodes` / `n_measurements` | Protocol geometry metadata |

---

**`.eit`-style but with a new extension (e.g. `.acm`)**

Add one branch in `detect_vendor_and_format()` that calls the same generic
`detect_vendor_from_eit_header()` (if the magic-string approach works for this format):

```python
# In detect_vendor_and_format() — detection.py:
if extension == ".acm":
    vendor = detect_vendor_from_eit_header(path)  # reuses HEADER_FORMAT_SPECS
    return FileDetection(path=path, extension=extension, vendor=vendor)
```

Then add the `HeaderFormatSpec` to `HEADER_FORMAT_SPECS` as above.

---

**Binary without a header (size-based, `.bin`)**

Add a `FormatSpec` to `BIN_FORMAT_SPECS` (see [section 2](#2-add-a-new-drger-bin-frame-size)).
No changes to `detection.py` needed.

---

**Tabular (CSV / TXT / ASC)**

File: `src/fasteit/parsers/detection.py` — extend `detect_vendor_from_tabular()`
with a keyword found in the first 40 lines:

```python
# In detect_vendor_from_tabular():
if "hamilton medical" in normalized:
    return "hamilton"
```

---

### Step 2 — Schema definition

**Binary** — define a numpy structured dtype (byte-exact) and a Medibus-style
field list. File: `src/fasteit/parsers/acmeit/acmeit_dtypes.py` (new)

```python
import numpy as np

ACMEIT_FRAME_DTYPE = np.dtype([
    ("ts",     "<f8"),           #    8 bytes
    ("pixels", "<f4", (32, 32)), # 4096 bytes
    ("flow",   "<f4"),           #    4 bytes
    ("paw",    "<f4"),           #    4 bytes
])
# Total: 8 + 4096 + 4 + 4 = 4112 bytes
```

**Tabular** — define a `(snake_case_name, unit, is_continuous)` field list.
*(Optional but recommended — makes the schema self-documenting and testable.)*
File: `src/fasteit/parsers/hamilton/hamilton_dtypes.py` (new)

```python
HAMILTON_FIELDS: list[tuple[str, str, bool]] = [
    ("flow",             "L/min", True),   # continuous waveform
    ("airway_pressure",  "mbar",  True),   # continuous waveform
    ("volume",           "mL",    True),   # continuous waveform
    ("respiratory_rate", "/min",  False),  # breath-averaged
    ("tidal_volume",     "mL",    False),  # breath-averaged
    ("peep",             "mbar",  False),  # breath-averaged
]
```

The `is_continuous` flag lets downstream code select only waveforms or only
breath-averaged values without touching the parser.

---

### Step 3 — Parser class

File: `src/fasteit/parsers/<vendor>/<vendor>_parser.py` (new)

The structure is always the same: `validate()` + `parse()`.

**Binary parser:**

```python
from pathlib import Path
from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.base import BaseParser

class AcmEitParser(BaseParser):

    def validate(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return False
        # e.g. check magic bytes or frame-size divisibility
        return True

    def parse(self, path: Path) -> ReconstructedFrameData:
        # np.memmap with ACMEIT_FRAME_DTYPE, sentinel → NaN, fs estimation ...
        return ReconstructedFrameData(
            frames=frames,
            aux_signals=aux_signals,
            fs=fs,
            filename=str(path),
            file_format="acm",
        )
```

**Tabular parser:**

```python
import io
import pandas as pd
from pathlib import Path
from fasteit.models.continuous_data import ContinuousSignalData
from fasteit.parsers.base import BaseParser
from fasteit.parsers.detection import detect_vendor_from_tabular

class HamiltonParser(BaseParser):

    def validate(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            return detect_vendor_from_tabular(path) == "hamilton"
        except ValueError:
            return False

    def parse(self, path: Path) -> ContinuousSignalData:
        path = Path(path)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Locate the column-header row (skip any prose header above it)
        header_idx = next(
            i for i, l in enumerate(lines)
            if l.strip().startswith("Time")  # adapt to real format
        )
        df = pd.read_csv(
            io.StringIO("".join(lines[header_idx:])),
            sep=";",            # or "\t" — check the real file
            decimal=",",
            na_values=["-", "---"],
            index_col=False,
        )
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        fs = None
        if "time" in df.columns:
            dt = pd.to_numeric(df["time"], errors="coerce").diff().dropna().median()
            if pd.notna(dt) and dt > 0:
                fs = round(1.0 / float(dt))  # time assumed already in seconds

        return ContinuousSignalData(
            table=df,
            fs=fs,
            filename=str(path),
            file_format=path.suffix.lstrip("."),
            metadata={"parsed_section": "ventilator_export"},
        )
```

---

### Step 4 — Register in loader

File: `src/fasteit/parsers/loader.py`

```python
from fasteit.parsers.hamilton.hamilton_parser import HamiltonParser   # or AcmEitParser

def default_parser_registry():
    return {
        ...
        ("hamilton", ".csv"): lambda: HamiltonParser(),
        ("hamilton", ".txt"): lambda: HamiltonParser(),
    }
```

Registry key = `(vendor_lowercase, extension_with_dot)`. If a vendor uses
multiple extensions for the same format, add one entry per extension pointing
to the same parser factory.

---

### Step 5 — Export from package `__init__`

File: `src/fasteit/parsers/__init__.py`

```python
from fasteit.parsers.hamilton.hamilton_parser import HamiltonParser
__all__ = [..., "HamiltonParser"]
```

---

### Note: same extension, different vendors

If two vendors share the same extension (e.g. both export `.csv`),
`detect_vendor_from_tabular()` reads the file content and returns a different
vendor string for each. The registry key `(vendor, extension)` ensures the
correct parser is called.

---

## 4. Testing checklist

Every new parser must have tests before it is considered done.
Minimal coverage for a tabular parser:

- [ ] `test_validate_returns_true_on_valid_file` — synthetic file, happy path
- [ ] `test_validate_returns_false_on_wrong_extension` — wrong extension
- [ ] `test_validate_returns_false_on_wrong_vendor_keyword` — wrong content
- [ ] `test_parse_returns_continuous_signal_data` — correct type and `file_format`
- [ ] `test_parse_fs_estimated` — `fs` is a reasonable integer (e.g. 50)
- [ ] `test_parse_column_names_normalised` — no spaces or special characters
- [ ] `test_parse_time_column_present` — `"time"` in `data.table.columns`
- [ ] `test_parse_continuous_channels_present` — at least one signal channel
- [ ] `test_parse_raises_on_missing_table_section` — `ValueError` on malformed file
- [ ] `test_load_data_routes_to_parser` — `load_data(path).vendor == "myvendor"`

---

## 5. File layout

```
src/fasteit/
├── models/
│   ├── base_data.py              BaseData — common fields for all containers
│   ├── reconstructed_data.py     ReconstructedFrameData — 32×32 pixel frames
│   ├── continuous_data.py        ContinuousSignalData — tabular signal export
│   └── raw_impedance_data.py     RawImpedanceData — 208 transimpedances per frame
│
└── parsers/
    ├── base.py                   BaseParser (validate / parse / parse_safe)
    ├── bin_formats.py            BIN_FORMAT_SPECS — all known .bin frame sizes
    ├── detection.py              detect_vendor_and_format() — routes any file
    ├── errors.py                 AmbiguousFormatError, UnknownFormatError
    ├── header_formats.py         EIT_HEADER_FORMATS — known .eit ASCII headers
    ├── loader.py                 load_data / load_many / load_folder
    │
    ├── draeger/
    │   ├── asc/
    │   │   └── asc_parser.py     DragerAscParser — .asc continuous export
    │   ├── bin/
    │   │   ├── bin_parser.py     DragerBinParser — .bin reconstructed frames
    │   │   ├── bin_utils.py      Sentinel → NaN, fs estimation helpers
    │   │   └── draeger_dtypes.py FRAME_BASE_DTYPE, FRAME_EXT_DTYPE, MEDIBUS_* fields
    │   └── eit/
    │       ├── eit_parser.py     DragerEitParser — .eit raw transimpedances
    │       ├── eit_dtypes.py     EIT_FRAME_DTYPE (5495-byte frame layout)
    │       ├── eit_utils.py      Calibration constants FT_A/FT_B/FC_CURRENT/FV_VOLTAGE
    │       └── eit_pyeit_bridge.py  build_greit() / reconstruct_greit() — pyEIT bridge
    │
    └── timpel/
        ├── timpel_parser.py      TimpelTabularParser — .csv/.txt reconstructed frames
        └── timpel_dtypes.py      TIMPEL_AUX_FIELDS — auxiliary signal columns
```

**Where to add a new vendor** — create `parsers/<vendor>/` mirroring the structure above:
one `<vendor>_parser.py`, one `<vendor>_dtypes.py`, one `__init__.py`.
Then follow steps 1–5 in [section 3](#3-add-a-new-parser).

---

