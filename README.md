# fastEIT

![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status: Pre-Alpha](https://img.shields.io/badge/Status-Pre--Alpha-orange)

Python library for parsing, preprocessing, and analyzing Electrical Impedance Tomography (EIT)
data from the Dräger PulmoVista 500. Extracts breath-level clinical features into
analysis-ready datasets for research in mechanical ventilation and lung monitoring.

## What makes this different

- **Native Dräger parser** — reads `.eit` (raw voltages) and `.bin` (reconstructed images) directly; no MATLAB required
- **Breath-level database** — one row per breath with 30+ EIT features, PEEP step ID, and quality flags; ready for R/Python/ML
- **Automatic PEEP step detection** — identifies PEEP trial steps from Medibus waveform or EELI signal
- **pyEIT compatible** — export to pyEIT format for offline image reconstruction

## Installation

```bash
git clone https://github.com/Kernel236/fastEIT.git
cd fastEIT
pip install -e ".[dev]"
```

## Project status

Fase 0 (scaffold) in progress.

## Disclaimer

This software is for **research purposes only**. It is not a medical device and has not been
validated for clinical decision-making. Use in clinical settings requires independent validation.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
