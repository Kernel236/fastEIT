# fastEIT - Review figures

Two figures to look at a recording before trusting its numbers (`fasteit.plotting`).
Colours and markers are the same in every figure.

## Before preprocessing: the segments

```python
from fasteit.models import Segment
from fasteit.plotting import plot_segmentation

segments = [
    Segment(0, 290, "phase", "tidal"),        # baseline
    Segment(320, 1500, "phase", "peep_trial"),
    Segment(600, 640, label="movement"),       # exclusion
]
plot_segmentation(load_data("P01_T0.bin"), segments, title="P01, T0")
```

The global signal of the file (sum of all pixels) with the segments as bands: phases
coloured by protocol (blue tidal, orange PEEP trial), exclusions grey and hatched, PEEP
steps placed by hand with their level. Below, the airway pressure in cmH2O when the file
has it.

## After preprocessing: the breaths

```python
from fasteit.plotting import plot_breaths

plot_breaths(trial)        # also a recording, TidalFeatures or PeepTrialFeatures
```

Each breath is a marker at its end-inspiratory peak on the signal the breaths were found
on:

| Marker | Breath |
|--------|--------|
| filled blue circle | used for the indices: the valid breaths of the phase (tidal), the breaths analysed in each step (PEEP trial) |
| empty blue circle | valid but not used: outside the phase, or in the first seconds of a step, or beyond the last 10 |
| orange triangle | flagged by a quality check |
| grey cross | excluded (the excluded parts are grey hatched bands) |

For a PEEP trial the steps are shaded and labelled with their PEEP level of the protocol
(`peep_set`, or the measured level without the protocol), and the PEEP of each breath is
drawn below with the level of each step. On the right, the tidal image inside the lung
mask (patient right on the left, ventral on top) and the count of each kind of breath.

Both functions return a matplotlib figure: `fig.savefig("P01_T0_breaths.png")` keeps one
image per recording next to the exported tables.
