# Install and supported files

## Install

```bash
pip install "fasteit @ git+https://github.com/Kernel236/fastEIT.git@v0.7.0a2"   # a fixed version
pip install git+https://github.com/Kernel236/fastEIT.git                        # the latest dev
```

Python 3.10 to 3.14. A study, or an app built on fastEIT, should install a fixed version: its results then do
not change when the library does. The versions are listed on the
[tags page](https://github.com/Kernel236/fastEIT/tags) and in the version menu of this site.

For development (tests, linting) and to build this site:

```bash
git clone https://github.com/Kernel236/fastEIT.git
cd fastEIT
pip install -e ".[dev,docs]"
mkdocs serve        # the site at http://127.0.0.1:8000/fastEIT/
```

## Supported files

The analysis starts from reconstructed images (32x32 pixels per frame):

| File | Read as | Analysed |
|------|---------|----------|
| Dräger PulmoVista 500 `.bin` | `ReconstructedFrameData` (images and ventilator signals) | yes |
| Timpel `.csv`, `.txt`, `.asc` export | `ReconstructedFrameData` | yes |
| Dräger `.eit` | `RawImpedanceData` (208 raw measurements per frame) | no: needs a reconstruction first |
| Dräger `.asc` export | `ContinuousSignalData` (signals only) | no: no images |

`load_data(path)` recognises the vendor and the format by itself. Large `.bin` and `.eit`
files are not read into memory at once: the images and measurements are read from disk
when they are used.

## The four steps

```python
from fasteit.export import recording_tables, write_csv
from fasteit.features import tidal_features
from fasteit.models import RecordingContext
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess

raw = load_data("recording.bin")                                   # 1. read the file
recording = preprocess(raw, context=RecordingContext(patient_id="P01", label="T0"))  # 2. breaths
features = tidal_features(recording)                               # 3. clinical indices
write_csv(recording_tables(features), "results/")                  # 4. tables for statistical analysis
```

Each step returns an object that keeps the one before it, with a `log` of what was done.
A PEEP trial adds one step (`segment_peep_trial`) between 2 and 3. The
user guide goes through the steps: [analyse a recording](guide/analyse.md) (both
protocols), [cut a recording](guide/segments.md), [review figures](review.md) and
[export the tables](export.md). How each layer works is under "How it works"
([data model](data_model.md)).

*Research use only - not a medical device.*
