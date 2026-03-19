# Changelog

All notable changes to fastEIT will be documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — 0.1.0.dev0

### Added (Fase 0 — Scaffold & Setup)
- Repository with Apache-2.0 license
- `src/fasteit/` package structure with placeholder modules:
  parsers, preprocessing, features, models, export, visualization, config
- `pyproject.toml` with hatchling build, core dependencies, `[dev]` and `[pyeit]` optional groups
- GitHub Actions: `tests` workflow (Python 3.10/3.11/3.12 matrix) and `lint` workflow (ruff)
- `docs/eit_format.md` — Dräger PulmoVista format specification (in progress)
- `docs/eidors_analysis.md` — EIDORS source analysis notes (in progress)
- `tests/test_placeholder.py` — basic import smoke test
- README with installation instructions and research disclaimer

### Added (Task 0.5.1 — BaseParser)
- `src/fasteit/parsers/base.py`: `BaseParser` ABC with `parse()`, `validate()` abstract methods
  and concrete `parse_safe()` template method (FileNotFoundError + ValueError on invalid input)
- `tests/test_base_parser.py`: 6 tests covering TypeError on ABC, FileNotFoundError, ValueError,
  successful parse, and str→Path coercion

### Added (Task 0.5.2 — BaseData + BinData)
- `src/fasteit/models/base_data.py`: `BaseData` dataclass — common fields for all format containers
  (filename, file_format, fs, metadata, n_frames, duration)
- `src/fasteit/models/bin_data.py`: `BinData(BaseData)` — structured numpy frame array with
  properties `timestamps`, `pixels`, `event_texts`, `global_signal`, `roi_signals`, `roi_signal()`
- `tests/test_bin_data.py`: 14 tests covering n_frames, duration, array shapes, values, ROI signals,
  custom fs, and error handling

### Added (Task 1.2.1 — Numpy dtypes for Dräger frame layouts)
- `src/fasteit/dtypes.py`: `FRAME_BASE_DTYPE` (4358 B) and `FRAME_EXT_DTYPE` (4382 B),
  little-endian structured dtypes for `.bin` frame parsing; `MEDIBUS_FIELDS` / `MEDIBUS_EXT_FIELDS`
  with units and continuity flags; `MEDIBUS_INDEX` / `MEDIBUS_EXT_INDEX` lookup dicts
- `src/fasteit/models/bin_data.py`: import dtype from `fasteit.dtypes`; new properties
  `min_max_flags` (breath phase markers) and `event_markers` (per-frame event counter);
  `timestamps` now reads `frames["ts"]` (float64 fraction-of-day, confirmed via eitprocessing)
- 39 tests passing (22 dtype + 17 BinData)

### Added (Task 0.6.1 partial — DeviceConfig)
- `src/fasteit/config.py`: `DeviceConfig` dataclass with PulmoVista hardware constants
  (fs=50.0 Hz, n_electrodes=16, n_measurements=208, pixel_grid=(32,32),
  frame_size_base=4358, frame_size_ext=4382)
- `PreprocessingConfig` and `AnalysisConfig` stubs (to be completed in Fase 4)
- `Config` aggregate dataclass (stub — fields to be populated incrementally)
