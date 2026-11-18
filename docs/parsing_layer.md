# fastEIT - Extending the parsers

How to add what the parsing layer does not read yet. How a file is recognised and read
is in [parsers](parsers.md); every step below plugs into that flow.

1. [Add a new Dräger `.bin` frame size](#1-add-a-new-dräger-bin-frame-size)
2. [Add a new parser](#2-add-a-new-parser)
3. [Testing checklist](#3-testing-checklist)
4. [Where a new device goes](#4-where-a-new-device-goes)

---

## 1. Add a new Dräger `.bin` frame size

If a new PulmoVista firmware introduces a new frame layout (e.g., 4406
bytes with 6 additional Medibus fields).

**Files to change: 2. Everything else is automatic.**

### Step 1 - Define the dtype

File: `src/fasteit/parsers/draeger/bin/draeger_dtypes.py`

Add the new structured dtype and its Medibus field list, following the existing ones.
The layout must add up exactly to the new frame size: check it with an assertion.

```python
# New extended frame (4406 bytes) - hypothetical example: the fields of the existing
# extended frame, with a larger Medibus block.
FRAME_NEWEXT_DTYPE = np.dtype([*FRAME_EXT_DTYPE.descr[:-1], ("medibus_data", "<f4", (64,))])
assert FRAME_NEWEXT_DTYPE.itemsize == 4406

MEDIBUS_NEWEXT_FIELDS: list[tuple[str, str, bool]] = _MEDIBUS_COMMON + [
    ("high_pressure", "mbar", False),  # (name, unit, is_continuous) of each new channel
    # ...
]
```

### Step 2 - Register the format spec

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
    ),
)
```

**That's it.** `detect_bin_format_from_size()` picks it up automatically.
`DragerBinParser.parse()` uses `spec.dtype` and `spec.medibus_fields` from
the spec - no parser changes needed.

---

## 2. Add a new parser

This recipe covers any combination: new vendor or existing vendor, binary or
tabular. The five steps are always the same; only the content of steps 1 and 2
differs by format type.

**Return type by format:**

| Format | Return type | Example |
|--------|-------------|---------|
| Binary frame sequence | `ReconstructedFrameData` | Dräger `.bin`, AcmEIT `.acm` |
| Tabular (CSV / TXT), signals only | `ContinuousSignalData` | Dräger `.asc`, Hamilton export, SERVO-U export, capnograph |
| Tabular with pixel columns | `ReconstructedFrameData` | Timpel `.csv` |

Tabular parsers are common for any device that exports to a plain-text table -
including standalone ventilators (Hamilton, Maquet SERVO, Dräger Evita) that
export continuous waveforms (Flow, Paw, Volume) and/or breath-averaged
parameters (TV, RR, PEEP, compliance) in the same file.

**Files to change: 4-5.**

---

### Step 1 - Detection

Detection works the same way for all three file families: **register a spec, the
generic function finds it automatically**. You rarely need to touch `detection.py`.

---

**`.eit`-style (header + binary frames, same `.eit` extension, new vendor)**

File: `src/fasteit/parsers/header_formats.py`

Just add a `HeaderFormatSpec` and append it to `HEADER_FORMAT_SPECS`.
`detect_vendor_from_eit_header()` iterates the list and returns the first match -
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
| `frame_size_bytes` | Byte-exact size of one binary frame - used by the parser |
| `n_electrodes` / `n_measurements` | Protocol geometry metadata |

---

**`.eit`-style but with a new extension (e.g. `.acm`)**

Add one branch in `detect_vendor_and_format()` that calls the same generic
`detect_vendor_from_eit_header()` (if the magic-string approach works for this format):

```python
# In detect_vendor_and_format() - detection.py:
if extension == ".acm":
    vendor = detect_vendor_from_eit_header(path)  # reuses HEADER_FORMAT_SPECS
    return FileDetection(path=path, extension=extension, vendor=vendor)
```

Then add the `HeaderFormatSpec` to `HEADER_FORMAT_SPECS` as above.

---

**Binary without a header (size-based, `.bin`)**

Add a `FormatSpec` to `BIN_FORMAT_SPECS` (see [section 2](#1-add-a-new-dräger-bin-frame-size)).
No changes to `detection.py` needed.

---

**Tabular (CSV / TXT / ASC)**

File: `src/fasteit/parsers/detection.py` - extend `detect_vendor_from_tabular()`
with a keyword found in the first 40 lines:

```python
# In detect_vendor_from_tabular():
if "hamilton medical" in normalized:
    return "hamilton"
```

---

### Step 2 - Schema definition

**Binary** - define a numpy structured dtype (byte-exact) and a Medibus-style
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

**Tabular** - define a `(snake_case_name, unit, is_continuous)` field list.
*(Optional but recommended - makes the schema self-documenting and testable.)*
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

### Step 3 - Parser class

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
        # np.memmap with ACMEIT_FRAME_DTYPE, fs estimation ...
        # (keep sentinel values as-is: NaN replacement belongs to preprocessing)
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
            sep=";",            # or "\t" - check the real file
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

### Step 4 - Register in loader

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

### Step 5 - Export from package `__init__`

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

## 3. Testing checklist

Every new parser must have tests before it is considered done.
Minimal coverage for a tabular parser:

- [ ] `test_validate_returns_true_on_valid_file` - synthetic file, happy path
- [ ] `test_validate_returns_false_on_wrong_extension` - wrong extension
- [ ] `test_validate_returns_false_on_wrong_vendor_keyword` - wrong content
- [ ] `test_parse_returns_continuous_signal_data` - correct type and `file_format`
- [ ] `test_parse_fs_estimated` - `fs` is a reasonable integer (e.g. 50)
- [ ] `test_parse_column_names_normalised` - no spaces or special characters
- [ ] `test_parse_time_column_present` - `"time"` in `data.table.columns`
- [ ] `test_parse_continuous_channels_present` - at least one signal channel
- [ ] `test_parse_raises_on_missing_table_section` - `ValueError` on malformed file
- [ ] `test_load_data_routes_to_parser` - `load_data(path).vendor == "myvendor"`

---

## 4. Where a new device goes

Create `parsers/<vendor>/` like the existing ones ([file layout](parsers.md#6-file-layout)):
one `<vendor>_parser.py`, one `<vendor>_dtypes.py`, one `__init__.py`. Then follow the
steps of [section 2](#2-add-a-new-parser).

*Research use only - not a medical device.*
