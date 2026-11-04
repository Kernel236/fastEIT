# fastEIT

| | |
|:---|:---|
| **Status** | ![Alpha](https://img.shields.io/badge/Status-Alpha-orange) |
| **License** | ![Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg) |
| **Python** | ![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue) |
| **Docs** | [![Docs](https://github.com/Kernel236/fastEIT/actions/workflows/docs.yml/badge.svg)](https://kernel236.github.io/fastEIT/) |
| **Coverage** | [![codecov](https://codecov.io/gh/Kernel236/fastEIT/branch/dev/graph/badge.svg)](https://codecov.io/gh/Kernel236/fastEIT) |
| **CI** | [![Tests](https://github.com/Kernel236/fastEIT/actions/workflows/test.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/test.yml) [![Lint](https://github.com/Kernel236/fastEIT/actions/workflows/lint.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/lint.yml) [![Build](https://github.com/Kernel236/fastEIT/actions/workflows/build.yml/badge.svg)](https://github.com/Kernel236/fastEIT/actions/workflows/build.yml) |

Python library for Electrical Impedance Tomography (EIT) recordings of the Dräger
PulmoVista 500 and of Timpel: it reads the files, finds and checks each breath, computes
the clinical indices of tidal recordings and PEEP trials, and writes them as tables for R,
SPSS and databases. For research in mechanical ventilation and lung monitoring.

**Documentation: [kernel236.github.io/fastEIT](https://kernel236.github.io/fastEIT/)**

## Install

```bash
pip install git+https://github.com/Kernel236/fastEIT.git
```

## From a file to the tables

```python
from fasteit.export import recording_tables, write_csv
from fasteit.features import tidal_features
from fasteit.models import RecordingContext
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess

raw = load_data("recording.bin")                                   # 1. read the file
recording = preprocess(raw, context=RecordingContext(patient_id="P01", label="T0"))  # 2. breaths
features = tidal_features(recording)                               # 3. clinical indices
write_csv(recording_tables(features), "results/")                  # 4. tables for R and SPSS
```

A PEEP trial adds `segment_peep_trial` between steps 2 and 3. The
[user guide](https://kernel236.github.io/fastEIT/) covers both protocols, cutting a
recording, the review figures and the export.

## What it does

| Step | |
|------|---|
| Files | Dräger `.bin` and Timpel exports (reconstructed images) are analysed; Dräger `.eit` (raw measurements) and `.asc` (signals) are read |
| Breaths | cardiac filter, breath detection with quality flags, exclusions, lung mask, regions; PEEP, plateau, driving pressure and tidal volume of each breath, in cmH2O and mL |
| PEEP trials | steps from the airway pressure or placed by hand, at the PEEP levels of the protocol; breaths analysed, driving pressure and tidal volume of each step |
| Clinical indices | EELI, TIV, share by region, global inhomogeneity, centre of ventilation, compliance; Costa collapse and overdistension, crossing PEEP, best PEEP, recruitment-to-inflation ratio |
| Tables | one row per recording, breath and PEEP step, keyed by patient, recording and phase; CSV for R, SPSS and Excel |
| Figures | the segments on the file, and the breaths used for the indices |

Every method states its source (paper and DOI, or an evidence notebook in `notebooks/`),
and the steps reimplemented from [eitprocessing](https://github.com/EIT-ALIVE/eitprocessing)
are checked against it.

## Status

Alpha: the pipeline from a file to the tables is complete and tested; names and tables
may still change before 1.0.

`.eit` files can be reconstructed with pyEIT through `reconstruct_greit()` (optional:
`pip install "fasteit[pyeit] @ git+https://github.com/Kernel236/fastEIT.git"`; GREIT,
Adler et al., *Physiol. Meas.* 2009, DOI: 10.1088/0967-3334/30/6/S03). This bridge is
experimental: it uses a generic circular mesh, and its images do not match the Dräger
reconstruction stored in `.bin` files.

**ML reconstruction (this branch, experimental).** `fasteit.reconstruction` learns the
Dräger reconstruction from paired `.eit`/`.bin` recordings with ridge regression (optional:
`fasteit[reconstruction]`): each recording is normalised to a reference at its lowest
end-expiration, the channels with a bad reference are set to 0, and the images are low-pass
filtered at 50/min as the Dräger ones. Only the method is published here: results on our data and
trained models stay private.

## Contributing

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). This project follows the
[Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

## Disclaimer

This software is for **research purposes only**. It is not a medical device and has not been
validated for clinical decision-making.

## License

Apache License 2.0 - see [LICENSE](LICENSE).
