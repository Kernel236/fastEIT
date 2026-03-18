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
