# fastEIT - Parsing Layer

The parsing layer (layer 1) reads a file as it is. It recognises the device and the
format by itself, chooses the parser and returns a data container, without changing a
value: the "no data" values, the units of the device and the timestamps stay as in the
file, and the preprocessing layer cleans them. Large `.bin` and `.eit` files are
memory-mapped: the frames are read from disk when they are used.

1. [Entry point](#1-entry-point)
2. [Steps](#2-steps)
3. [Output](#3-output)
4. [More than one file](#4-more-than-one-file)
5. [The `.eit` file](#5-the-eit-file)
6. [File layout](#6-file-layout)

---

## 1. Entry point

```python
from fasteit.parsers.loader import load_data

data = load_data("patient01.bin")   # Dräger .bin  -> ReconstructedFrameData
data = load_data("patient01.eit")   # Dräger .eit  -> RawImpedanceData
data = load_data("export.asc")      # Dräger .asc  -> ContinuousSignalData
```

`load_data(path)` is the only function needed: the caller never chooses the parser. A
file that cannot be read stops with an error that says why:

| Problem | Error |
|---------|-------|
| The file does not exist | `FileNotFoundError` |
| Extension not supported, device not recognised, or the file fails the check of its parser | `ValueError` |
| A `.bin` whose size fits no known frame layout, or more than one | `UnsupportedFrameSizeError`, `AmbiguousFormatError` (both `ParserError`) |
| No parser for the device and extension found | `NotImplementedError` |

## 2. Steps

```
load_data(path)
 ├─ 1. detect_vendor_and_format(path)      what file is this?
 ├─ 2. build_parser_from_detection(...)    which parser reads it?
 ├─ 3. parser.parse_safe(path)             check the file, then read it
 └─ 4. what was detected goes in data.metadata
```

| # | Step | Module |
|---|------|--------|
| 1 | Recognise the file. The extension says the family, the content says the device: a `.bin` has no header, so its layout is the frame size that divides the file size exactly (one layout fits: that one; none or two: an error); a `.eit` is recognised by a text string at the start of its header; a `.csv`, `.txt` or `.asc` by its first 40 lines (`DraegerEIT Software` for Dräger; `timpel`, or a first row of 1030 values, for Timpel) | `detection` |
| 2 | Choose the parser in the registry, by device and extension. Spellings of the device (`Drager`, `Dräger`, `DRAEGER`) are the same device | `loader` |
| 3 | `parse_safe`: the parser checks the file (`validate`), then reads it (`parse`) | `base` and the parser of the format |
| 4 | The device, the extension and, for a `.bin`, the frame layout found are written in `data.metadata` (`detected_vendor`, `detected_extension`, `detected_bin_format`) | `loader` |

The same extension can belong to two devices (a `.csv` from Dräger or from Timpel): the
content decides, and the pair (device, extension) chooses the parser. A Dräger `.bin`
comes in two layouts, with and without the PressurePod channels; the channels are in the
file whether or not a Pod was connected, and `data.metadata["has_pressure_pod"]` says
whether they hold values.

Reading a Dräger `.bin`, step by step:

```
load_data("patient01.bin")
 ├─ detect_vendor_and_format()     the file size is a whole number of frames
 │                                 of one known layout -> Dräger, that layout
 ├─ build_parser_from_detection()  -> DragerBinParser
 ├─ parse_safe()
 │   ├─ validate()                 the size still fits the layout
 │   └─ parse()                    frames memory-mapped, sampling frequency from
 │                                 the timestamps -> ReconstructedFrameData
 └─ metadata: detected_vendor, detected_extension, detected_bin_format
```

**Sampling frequency.** A Dräger `.bin` does not store it: it is estimated from the
timestamps (50 Hz, with a note in `metadata["warnings"]`, when that fails). A `.eit`
has it in its header; a Dräger `.asc` gets it from its time column (none without one); a
Timpel export has no time column, so it is 50 Hz.

## 3. Output

| File | Parser | Container | What it holds |
|------|--------|-----------|---------------|
| Dräger PulmoVista 500 `.bin` | `DragerBinParser` | `ReconstructedFrameData` | 32x32 images reconstructed by the device, timestamps, the device markers of each breath and the event texts, the ventilator (Medibus) and PressurePod signals |
| Timpel `.csv`, `.txt`, `.asc` | `TimpelTabularParser` | `ReconstructedFrameData` | 32x32 images, airway pressure, flow, volume and the device markers |
| Dräger `.eit` | `DragerEitParser` | `RawImpedanceData` | 208 calibrated transimpedances per frame (16 electrodes, adjacent drive) and the ventilator signals: no images |
| Dräger `.asc` | `DragerAscParser` | `ContinuousSignalData` | the signals exported by the device software (global and regional impedance, markers, Medibus): no images |

Only `ReconstructedFrameData` goes on to `preprocess()`. The fields of each container
are in the [data model](data_model.md).

## 4. More than one file

`load_many(paths)` reads a list of files, in order, with `load_data()`; devices and
formats can be mixed. `load_folder(folder, pattern="*")` reads every file of a folder,
sorted by name, and skips a file it cannot read with a one-line message on `stderr`, so
one bad file does not stop the others.

```python
from fasteit.parsers.loader import load_folder, load_many

recordings = load_many(["patient01.bin", "patient02.bin"])
all_bin = load_folder("/data/session01/", pattern="*.bin")
recursive = load_folder("/data/", pattern="**/*.bin")
```

All three functions take an optional `registry=`, to add a parser without changing the
library ([extending the parsers](parsing_layer.md)).

## 5. The `.eit` file

A Dräger `.eit` holds the raw measurements of the belt, not images. fastEIT reads it
(we are not aware of another open-source Python reader), but the analysis starts from
images. As an example of what the measurements allow,
`fasteit.parsers.draeger.eit.eit_pyeit_bridge` reconstructs them with GREIT through pyEIT
(`pip install "fasteit[pyeit]"`; Adler et al., *Physiol. Meas.* 2009;30:S35-S55, DOI:
[10.1088/0967-3334/30/6/S03](https://doi.org/10.1088/0967-3334/30/6/S03)). It uses a
generic circular mesh, not the device's: its images are not those of the `.bin`, and
the pipeline does not use them.

## 6. File layout

```
src/fasteit/
├── models/
│   ├── base_data.py              BaseData: fields common to every container
│   ├── reconstructed_data.py     ReconstructedFrameData: images
│   ├── continuous_data.py        ContinuousSignalData: signals only
│   └── raw_impedance_data.py     RawImpedanceData: raw measurements
└── parsers/
    ├── loader.py                 load_data, load_many, load_folder, the registry
    ├── detection.py              detect_vendor_and_format: what file is this?
    ├── base.py                   BaseParser: validate, parse, parse_safe
    ├── bin_formats.py            the known .bin frame layouts
    ├── header_formats.py         the known .eit headers
    ├── errors.py                 ParserError and its subclasses
    ├── draeger/
    │   ├── bin/                  DragerBinParser, frame layouts, timestamps
    │   ├── eit/                  DragerEitParser, header and calibration, GREIT bridge
    │   └── asc/                  DragerAscParser
    └── timpel/                   TimpelTabularParser, auxiliary columns
```

*Research use only - not a medical device.*
