# Contributing to fastEIT

Every contribution is welcome — whether it is a bug report, a new parser,
a documentation fix, or just a question. This is an open project and there
is no wrong way to get involved, as long as you bring curiosity and
a collaborative attitude. If you are unsure whether your idea fits, open an
issue and let's talk about it.

## Reporting bugs

Open a GitHub issue with:
- Python version and OS
- Minimal code that reproduces the problem
- Full traceback

## Requesting features

Open a GitHub issue describing the use case and the expected behaviour.

## Development setup

```bash
git clone https://github.com/Kernel236/fastEIT.git
cd fastEIT
pip install -e ".[dev]"
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Both checks run automatically on every push via GitHub Actions.

## Running tests

```bash
pytest tests/
```

Tests that require real patient files are skipped automatically if the files
are not present (they are gitignored).

## Adding a new parser

See [docs/parsing_layer.md](docs/parsing_layer.md) for a step-by-step recipe
covering detection, schema definition, parser class, loader registration, and
the testing checklist.

## Pull request process

1. Fork the repository and create a branch from `dev`
2. Make your changes with tests
3. Ensure `ruff check` and `pytest` pass locally
4. Open a PR targeting `dev` (not `main`)
5. Describe what changed and why

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
