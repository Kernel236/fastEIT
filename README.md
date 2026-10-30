# fastEIT

| | |
|:---|:---|
| **Status** | ![Pre-Alpha](https://img.shields.io/badge/Status-Pre--Alpha-orange) |
| **License** | ![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg) |
| **Python** | ![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue) |
| **Coverage** | [![codecov](https://codecov.io/gh/Kernel236/fastEIT/branch/dev/graph/badge.svg)](https://codecov.io/gh/Kernel236/fastEIT) |
| **CI / CD** | [![Tests](https://github.com/Kernel236/fastEIT/actions/workflows/test.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/test.yml) [![Lint](https://github.com/Kernel236/fastEIT/actions/workflows/lint.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/lint.yml) [![Build](https://github.com/Kernel236/fastEIT/actions/workflows/build.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/build.yml) ![deploy](https://img.shields.io/badge/deploy-todo-lightgrey) |

Python library for parsing and preprocessing Electrical Impedance Tomography (EIT) data
from the Dräger PulmoVista 500. Provides structured data containers and a breath-by-breath
preprocessing for research in mechanical ventilation and lung monitoring.

## Installation

**Users:**

```bash
pip install git+https://github.com/Kernel236/fastEIT.git
```

**Developers:**

```bash
git clone https://github.com/Kernel236/fastEIT.git
cd fastEIT
pip install -e ".[dev]"
```

## Quick start

```python
from fasteit.parsers.loader import load_data

# Dräger .bin - reconstructed 32×32 frames + Medibus signals
data = load_data("patient01.bin")
print(data.n_frames, data.fs)       # e.g. 11500, 50.0
print(data.pixels.shape)            # (11500, 32, 32)
print(data.global_signal[:10])      # first 10 frames global EIT signal

# Dräger .eit - raw transimpedances for pyEIT reconstruction
data = load_data("patient01.eit")
print(data.n_frames, data.fs)          # e.g. 11500, 50.0
print(data.measurements.shape)        # (11500, 208) - 208 = 16 inj × 13 meas
print(list(data.aux_signals.keys()))   # timestamp, trans_A, trans_B, I_real, V_diff, medibus, ...

# Dräger .asc - continuous frame-by-frame signal export
data = load_data("patient01.asc")
print(data.n_frames, data.fs)       # e.g. 11500, 50.0
print(data.table.columns.tolist())  # all available signal columns

# Timpel .csv - reconstructed frames + aux signals
data = load_data("recording.csv")
print(data.n_frames, data.fs)       # e.g. 9000, 50.0
print(data.pixels.shape)            # (9000, 32, 32)
```

## Preprocessing

`preprocess()` turns a recording with pixels (Dräger `.bin`, Timpel) into an
`EITRecording` ready for breath-by-breath analysis: cardiac filter, breaths with
quality flags, exclusions, lung mask, regions of interest and regional signals.

```python
from fasteit.config import PreprocessingConfig
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess

raw = load_data("patient01.bin")
recording = preprocess(
    raw,
    config=PreprocessingConfig(heart_rate_bpm=85),  # optional: enables the MDN filter
    exclusions=[Segment(120, 180, label="cough")],  # optional: parts to exclude
    context=RecordingContext(patient_id="PZ01", position="supine"),
)
recording.breaths            # one row per breath: times, quality flags, is_valid
recording.lung_mask          # (32, 32) ventilated lung
recording.roi_signals        # ventral/dorsal, layers, right/left, quadrants
recording.aux                # pressures, flow, device markers ("no data" as NaN)
print("\n".join(recording.log))  # what each step did, with warnings
```

The steps reimplement [eitprocessing](https://github.com/EIT-ALIVE/eitprocessing)
v1.8.8 (Somhorst et al., *JOSS* 2026, DOI: 10.21105/joss.08179) and are checked
against it; the evidence for every choice is in
`notebooks/05_preprocessing_choices.ipynb`. See
[`docs/preprocessing.md`](docs/preprocessing.md).

## Project status

Pre-alpha. Data model, parsing layer and preprocessing layer implemented.

**Data model** (`models/`):

| Class | Content | Produced by |
|-------|---------|-------------|
| `ReconstructedFrameData` | 32×32 pixel matrices + synchronized signals | Dräger `.bin`, Timpel `.csv`/`.txt`/`.asc` |
| `ContinuousSignalData` | Signal table, one row per frame | `.asc` |
| `RawImpedanceData` | Calibrated transimpedances + aux signals for pyEIT | `.eit` |
| `EITRecording` | Preprocessed recording: breaths, lung mask, regions, log | `preprocess()` |

**Parsers** (`parsers/`):

| Parser | Status | Formats |
|--------|--------|---------|
| `DragerBinParser` | Implemented | `.bin` - base frame (4358 b) and PressurePod frame (4382 b); registry-driven, new frame sizes require only a dtype + one entry |
| `DragerAscParser` | Implemented | `.asc`, `.txt`, `.csv` - continuous waveform export |
| `DragerEitParser` | Implemented | `.eit` - ASCII header + binary frames (5495 b/frame); 208 calibrated transimpedances + Medibus aux signals |
| `TimpelTabularParser` | Implemented | `.csv`, `.txt`, `.asc` - reconstructed frame export |

All parsers are accessible via the single entry point `load_data(path)`, which
auto-detects vendor and format.

`RawImpedanceData` from `.eit` files can be passed to pyEIT through
`reconstruct_greit()` (optional dependency: `pip install fasteit[pyeit]`; implements
GREIT - Adler et al., *Physiol. Meas.* 2009, DOI: 10.1088/0967-3334/30/6/S03).
This function is far from reliable: it was written only to show one possible
application of the `.eit` parser. It uses a generic circular mesh, so the images do
not match the Dräger reconstruction stored in `.bin` files.

```python
from fasteit.parsers.draeger.eit.eit_pyeit_bridge import reconstruct_greit

data = load_data("patient01.eit")
images = reconstruct_greit(data.measurements)  # (N_frames, 32, 32)
```

## Documentation

- [`docs/data_model.md`](docs/data_model.md) - data container field specifications
- [`docs/parsers.md`](docs/parsers.md) - parser reference: formats, calibration, GREIT bridge, loader utilities
- [`docs/parsing_layer.md`](docs/parsing_layer.md) - architecture and extension recipes
- [`docs/preprocessing.md`](docs/preprocessing.md) - preprocessing steps, outputs, logs and limits

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
report bugs, request features, and submit pull requests.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## Disclaimer

This software is for **research purposes only**. It is not a medical device and has not been
validated for clinical decision-making. Use in clinical settings requires independent validation.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
