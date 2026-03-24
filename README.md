# fastEIT

| | |
| **Status** | ![Pre-Alpha](https://img.shields.io/badge/Status-Pre--Alpha-orange) |
| **License** | ![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg) |
| **Python** | ![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue) |
| **Coverage** | [![codecov](https://codecov.io/gh/Kernel236/fastEIT/branch/main/graph/badge.svg)](https://codecov.io/gh/Kernel236/fastEIT) |
| **CI / CD** | [![Tests](https://github.com/Kernel236/fastEIT/actions/workflows/test.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/test.yml) [![Lint](https://github.com/Kernel236/fastEIT/actions/workflows/lint.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/lint.yml) [![Build](https://github.com/Kernel236/fastEIT/actions/workflows/build.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/build.yml) ![deploy](https://img.shields.io/badge/deploy-todo-lightgrey) |

Python library for parsing Electrical Impedance Tomography (EIT) data from the Dräger
PulmoVista 500. Provides structured data containers for research in mechanical ventilation and lung monitoring.

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

# Dräger .bin — reconstructed 32×32 frames + Medibus signals
data = load_data("patient01.bin")
print(data.n_frames, data.fs)       # e.g. 11500, 50.0
print(data.pixels.shape)            # (11500, 32, 32)
print(data.global_signal[:10])      # first 10 frames global EIT signal

# Dräger .asc — continuous frame-by-frame signal export
data = load_data("patient01.asc")
print(data.n_frames, data.fs)       # e.g. 11500, 50.0
print(data.table.columns.tolist())  # all available signal columns

# Timpel .csv — reconstructed frames + aux signals
data = load_data("recording.csv")
print(data.n_frames, data.fs)       # e.g. 9000, 50.0
print(data.pixels.shape)            # (9000, 32, 32)
```

## Project status

Pre-alpha. Data model and parsing layer implemented.

**Data model** (`models/`):

| Class | Content | Produced by |
|-------|---------|-------------|
| `ReconstructedFrameData` | 32×32 pixel matrices + synchronized signals | `.bin`, `.txt` |
| `ContinuousSignalData` | Signal table, one row per frame | `.asc` |
| `RawImpedanceData` | Raw transimpedances for pyEIT | `.eit` (scaffold) |

**Parsers** (`parsers/`):

| Parser | Status | Formats |
|--------|--------|---------|
| `DragerBinParser` | Implemented | `.bin` — base frame (4358 b) and PressurePod frame (4382 b); registry-driven, new frame sizes require only a dtype + one entry |
| `DragerAscParser` | Implemented | `.asc`, `.txt`, `.csv` — continuous waveform export |
| `DragerEitParser` | Scaffold | `.eit` (Fase 2) |
| `TimpelTabularParser` | Implemented | `.csv`, `.txt` — reconstructed frame export |

All parsers are accessible via the single entry point `load_data(path)`, which
auto-detects vendor and format.

## Documentation

- [`docs/data_model.md`](docs/data_model.md) — data containers and parsing flow per file type
- [`docs/parsing_layer.md`](docs/parsing_layer.md) — how to extend: new frame sizes, new vendors, new formats

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
report bugs, request features, and submit pull requests.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## Disclaimer

This software is for **research purposes only**. It is not a medical device and has not been
validated for clinical decision-making. Use in clinical settings requires independent validation.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
