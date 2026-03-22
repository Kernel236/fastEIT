# fastEIT — Parsing Layer: How It Works and How to Extend It

This document explains the architecture of the parsing layer and provides
step-by-step recipes for the most common extension tasks:

1. [How the parsing layer works](#1-how-the-parsing-layer-works)
2. [Add a new Dräger `.bin` frame size](#2-add-a-new-drger-bin-frame-size)
3. [Add a new vendor from scratch](#3-add-a-new-vendor-from-scratch)
4. [Add a tabular format for an existing vendor](#4-add-a-tabular-format-for-an-existing-vendor)
5. [Testing checklist](#5-testing-checklist)

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
| `.eit` | header-based | reads first bytes, looks for ASCII magic string |
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
    ("draeger", ".bin"):  lambda: DragerBinParser(),
    ("draeger", ".asc"):  lambda: DragerAscParser(),
    ("draeger", ".txt"):  lambda: DragerAscParser(),
    ("draeger", ".csv"):  lambda: DragerAscParser(),
    ("timpel",  ".csv"):  lambda: TimpelTabularParser(),
    ...
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
    │       ├─ sentinel → NaN substitution
    │       ├─ fs estimation from timestamps
    │       └─ ReconstructedFrameData(frames, aux_signals, fs, ...)
    │
    └─ attach detected vendor/extension to data.metadata
       return ReconstructedFrameData
```

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

## 3. Add a new vendor from scratch

Use case: support files from a hypothetical "AcmEIT" device that produces
`.acm` binary files with its own frame layout.

**Files to change: 5.**

### Step 1 — Detection

File: `src/fasteit/parsers/detection.py`

For header-based detection, add a branch in `detect_vendor_and_format()`:

```python
if extension == ".acm":
    vendor = detect_vendor_from_acm_header(path)
    return FileDetection(path=path, extension=extension, vendor=vendor)
```

And the detection function itself:

```python
def detect_vendor_from_acm_header(path: Path) -> str:
    """Detect vendor from .acm file magic bytes."""
    with path.open("rb") as f:
        magic = f.read(8)
    if magic == b"ACMEIT\x01\x00":
        return "acmeit"
    raise ValueError(f"Unrecognised .acm header in '{path}'.")
```

For size-based binary formats (like `.bin`), add a `FormatSpec` to
`BIN_FORMAT_SPECS` instead — no detection function needed.

### Step 2 — Dtype (if binary)

File: `src/fasteit/parsers/acmeit/acmeit_dtypes.py` (new file)

```python
import numpy as np

ACMEIT_FRAME_DTYPE = np.dtype([
    ("ts",     "<f8"),
    ("pixels", "<f4", (32, 32)),
    ("flow",   "<f4"),
    ("paw",    "<f4"),
])
# Total: 8 + 4096 + 4 + 4 = 4112 bytes
```

### Step 3 — Parser

File: `src/fasteit/parsers/acmeit/acmeit_parser.py` (new file)

```python
from pathlib import Path
from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.base import BaseParser

class AcmEitParser(BaseParser):

    def validate(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return False
        # Add any format-specific checks here
        return True

    def parse(self, path: Path) -> ReconstructedFrameData:
        ...
        return ReconstructedFrameData(
            frames=frames,
            aux_signals=aux_signals,
            fs=fs,
            filename=str(path),
            file_format="acm",
        )
```


### Step 4 — Register in loader

File: `src/fasteit/parsers/loader.py`

```python
from fasteit.parsers.acmeit.acmeit_parser import AcmEitParser

def default_parser_registry():
    return {
        ...
        ("acmeit", ".acm"): lambda: AcmEitParser(),
    }
```

### Step 5 — Export from package `__init__`

File: `src/fasteit/parsers/__init__.py`

```python
from fasteit.parsers.acmeit.acmeit_parser import AcmEitParser
__all__ = [..., "AcmEitParser"]
```

---

## 4. Add a tabular format for an existing vendor

Use case: Dräger introduces a new `.csv` export with a different column schema
than the `.asc` — or you want a second parser for the `.asc` that also reads
the breath-averaged Tidal Variations section.

The simplest path is to create a second parser class and add a new registry
key. Registry keys are `(vendor, extension)` tuples — if the extension is the
same but the internal structure is different, you need detection logic to
choose between parsers before registering.

**If the extension is new (e.g., `.draegercsv`):**

1. Add detection in `detect_vendor_and_format()` for the new extension.
2. Create `parsers/draeger/csv/csv_parser.py` with `DragerCsvParser(BaseParser)`.
3. Add `("draeger", ".draegercsv"): lambda: DragerCsvParser()` to the registry.

**If the extension is the same but content differs:**

Detection must inspect file content and return a different vendor string or a
sub-format indicator. Then register two keys pointing to two parsers, and have
the detection logic choose. Alternatively, add an optional `parse_mode`
parameter to the existing parser.

---
