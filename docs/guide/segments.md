# Cut a recording

Parts of a recording are described by time intervals (`Segment`), in seconds from the
start of the file. There are three kinds:

| Kind | What it does | Example |
|------|--------------|---------|
| exclusion | its breaths are never analysed (movement, suction, recalibration) | `Segment(600, 640, label="movement")` |
| phase | a part with its own protocol, analysed on its own | `Segment(0, 290, "phase", "tidal")` |
| PEEP step placed by hand | the breaths of a step and its PEEP level, when there is no pressure signal or to choose the breaths | `Segment(60, 100, category="peep_step", parameters={"peep": 16})` |

## 1. Find the times

Look at the whole recording before cutting it: the global signal and the airway pressure,
with the segments drawn on top.

```python
from fasteit.models import Segment
from fasteit.parsers.loader import load_data
from fasteit.plotting import plot_segmentation

raw = load_data("P02_T0.bin")
plot_segmentation(raw)                       # read the times on the x axis

segments = [
    Segment(0, 290, "phase", "tidal"),
    Segment(300, 1500, "phase", "peep_trial"),
    Segment(285, 300, label="movement"),
]
plot_segmentation(raw, segments)             # check the cuts before preprocessing
```

The times read on the figure are precise enough: what matters is which breaths fall
inside, not the exact sample.

## 2. Use them

```python
from fasteit.features import tidal_features
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

recording = preprocess(raw, exclusions=[s for s in segments if s.category == "exclusion"])
baseline = tidal_features(recording, phase=segments[0])
trial = segment_peep_trial(recording, phase=segments[1])
```

Steps placed by hand go to `segment_peep_trial(recording, peep_steps=[...])`, with the
PEEP of each step and, if known, its plateau or driving pressure and tidal volume in
`parameters`.

## Rules

- **A breath belongs to an interval when it starts and ends inside it**: a breath cut by
  the edge of a phase or of a step is left out.
- **Exclusions are widened by 0.5 s on each side** (filters spread an artifact over the
  next samples), and every breath that touches them is excluded. To exclude one breath
  only, use a short interval around its peak, not the whole breath.
- **Excluding is not cutting**: the signal stays whole, with its original times, and the
  excluded breaths stay in the tables marked `excluded`.
- **A step placed by hand is the selection of breaths to analyse**: all its valid breaths,
  for example the ten chosen on the figure.
- **One recording is one file.** Two parts of the same file share the preprocessing
  (lung mask, filters); two files of the same patient are two recordings, each with its
  own.

Details: [preprocessing](../preprocessing.md) (exclusions) and
[segmentation](../segmentation.md) (phases and steps).
