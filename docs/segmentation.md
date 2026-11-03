# fastEIT - Segmentation Layer (PEEP trials)

The segmentation layer (layer 2b) takes a preprocessed recording (`EITRecording`) and
splits it by protocol. For a PEEP trial it finds the PEEP steps from the PEEP of each
breath, chooses the breaths analysed in each step and gives each step its driving
pressure and tidal volume. It is still preparation of the data: the clinical indices
(EELI, TIV, compliance, Costa) come in layer 3 ([`features.md`](features.md)).

The PEEP, plateau and tidal volume of each breath are read in layer 2
([`preprocessing.md`](preprocessing.md), section 3), in cmH2O and mL for every
vendor. The evidence for the pressure readings and the step detection is in
`notebooks/06_peep_trial_segmentation.ipynb` (four decremental PEEP trials from two
patients). The 15 s left out after each change, the 10 (at least 5) breaths analysed
and the 0.5 cmH2O tolerance are protocol choices.

1. [Entry point](#1-entry-point)
2. [Steps](#2-steps)
3. [Driving pressure and tidal volume of each step](#3-driving-pressure-and-tidal-volume-of-each-step)
4. [Recordings with more than one protocol](#4-recordings-with-more-than-one-protocol)
5. [Output](#5-output)
6. [Limits](#6-limits)

---

## 1. Entry point

```python
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

context = RecordingContext(protocol="peep_trial", tidal_volume_ml=450)  # Vt optional
recording = preprocess(load_data("peep_trial.bin"), context=context)
trial = segment_peep_trial(recording)

trial.steps              # one row per PEEP step: level, breaths, driving pressure, Vt
trial.analysed_breaths   # the breaths to analyse, with PEEP, plateau, driving pressure
print("\n".join(trial.log))
```

Without a pressure signal (or to choose the breaths yourself) the steps are placed by
hand: for each step, a segment around the breaths to analyse (e.g. the 10 stable breaths
chosen on the figure), with the PEEP level of the step (cmH2O):

```python
steps = [
    Segment(60, 100, category="peep_step", parameters={"peep": 16}),
    Segment(110, 150, category="peep_step", parameters={"peep": 14}),
]
trial = segment_peep_trial(recording, peep_steps=steps)
```

Steps placed by hand replace the automatic ones, and a segment is the selection of
breaths to analyse: every valid breath entirely inside it is analysed, with no time left
out after the change of PEEP and no limit of 10 (those rules apply only to the steps found
from the pressure). A breath cut by the edge of the segment, or with a quality flag, is
left out: draw the segment a little wider than the breaths chosen. An inspiratory hold is
looked for only inside the segment: include its breath, or type the plateau or the
driving pressure.
Without a pressure signal the plateau read on the ventilator can be typed with the
step, `parameters={"peep": 16, "plateau": 28}`, or directly the driving pressure,
`parameters={"peep": 16, "driving_pressure": 12}` (e.g. in pressure control, where it
is set), and so can the tidal volume, `parameters={"peep": 16, "tidal_volume": 450}`.
When there is a pressure signal, a typed PEEP more than 1 cmH2O from the measured one
gives a warning. Parameters are in `SegmentationConfig`.

## 2. Steps

| # | Step | Module |
|---|------|--------|
| 1 | Breaths of the PEEP trial phase, with the PEEP and plateau of each breath read in layer 2 | `pipeline` |
| 2 | Steps: a new step when 3 breaths in a row are more than 1 cmH2O (half of the step of the protocol) from the current level and within 1 cmH2O of each other; level = median PEEP. Excluded breaths (maneuvers) do not count | `peep_steps` |
| 3 | Breaths analysed: the first 15 s after each change are left out; then the last valid breaths whose PEEP is within 0.5 cmH2O of the step level (a breath without a PEEP reading is not analysed), up to 10; fewer than 5 give a warning | `peep_steps` |
| 4 | Step table: level of the protocol, driving pressure and tidal volume of each step | `pipeline` |

**Level of the protocol** (`peep_set`). The PEEP trial follows the protocol, the same for
every patient, while the measured level is noisy and in cmH2O (10 mbar set on a Dräger
ventilator is 10.2 cmH2O). Each step gets the nearest PEEP level of the protocol, given
as `SegmentationConfig(peep_levels=(16, 14, 12, 10, 8, 6, 4, 2, 0))`, after removing the
offset common to all steps (in a trial of notebook 06 every step read 0.6 cmH2O above the
set level, and 1.1 cmH2O at PEEP 0, nearer to 2 than to 0). A step more than
`step_threshold` (half of the step of the protocol) from every level gets none, with a
warning; without the levels of the protocol `peep_set` is empty. `peep_set` is the level to compare
and average steps between patients; `peep` stays the measured one.

The event texts typed on the device are not used to cut the steps: in the trials of the
notebook they were written from 2 s after to 14 s before the change of pressure.

## 3. Driving pressure and tidal volume of each step

The driving pressure of each step (the input of compliance in layer 3) works with the
usual protocols:

| Protocol | Driving pressure of the step (`driving_pressure_source`) |
|----------|-------------------------------------------------------|
| Volume control with a continuous end-inspiratory pause | median of the analysed breaths (`"breaths"`) |
| Volume control with one inspiratory hold per step | median of the breaths with a hold, one is enough (`"hold"`) |
| Pressure control | the flat inspiratory pressure is read as plateau (`"breaths"`), valid if the flow is zero at the end of inspiration |
| No pressure signal | typed with the step placed by hand (`"typed"`) |

The tidal volume of each step turns compliance into mL/cmH2O and EELI changes into
mL. In order (`tidal_volume_source`):

| Source | How | `tidal_volume_source` |
|--------|-----|-----------------------|
| Typed for each step | in the segment of a step placed by hand, or `segment_peep_trial(recording, tidal_volumes=[450, 460, None, ...])`, one per step in time order (e.g. the mean of each step on the case report form) | `"typed"` |
| Typed for the recording | `RecordingContext(tidal_volume_ml=450)`: the volume set in volume control, only when it is the same in every step (in pressure control it changes with PEEP: type it per step) | `"typed_recording"` |
| Measured | median of the analysed breaths, from the volume or flow curve of the ventilator (layer 2) | `"breaths"` |

Without any of them the tidal volume is missing and compliance stays in EIT units. A
typed volume outside 100-1500 mL, or more than 15 % from the one measured on the
analysed breaths of the step, gives a warning: a typing
error, or one volume typed for a trial in pressure control. How the tidal volume is
used in layer 3, and the assumptions of each use (compliance of the respiratory
system, volume of a region, change of EELI in mL), are in [`features.md`](features.md).

## 4. Recordings with more than one protocol

A recording can contain, for example, a tidal baseline and then a PEEP trial. Each part
is a **phase** with its own protocol, and the segmentation is limited to the PEEP trial
phase:

```python
phase = Segment(150, 600, category="phase", label="peep_trial")
trial = segment_peep_trial(recording, phase=phase)
```

Breaths outside the phase (or cut by its edges) have `peep_step = -1` and are never
analysed. A part with no planned analysis (for example an airway opening pressure
maneuver) is simply an exclusion of `preprocess()`: its breaths are never analysed and do
not count when the steps are found. Without a phase the whole recording is segmented.

## 5. Output

`PeepTrial` fields:

| Field | Content |
|-------|---------|
| `recording`, `config` | the preprocessed recording and the parameters used |
| `source` | `"pressure"` or `"manual"` |
| `phase` | the PEEP trial part of the recording, or None for the whole recording |
| `steps` | `step`, `peep` (level), `peep_set` (level of the protocol), `start_time`, `end_time`, `n_breaths`, `n_analysed`, `enough`, `plateau`, `driving_pressure`, `driving_pressure_source`, `n_driving_pressure`, `enough_driving_pressure`, `tidal_volume`, `tidal_volume_source` |
| `breaths` | the breath table of the recording (with `peep`, `plateau`, `driving_pressure`, `tidal_volume`) plus `peep_step`, `analysed` |
| `log` | what the segmentation did, with warnings |

Pressures are in cmH2O and volumes in mL for every vendor (`preprocess()` converts the
Dräger mbar, 1 mbar = 1.02 cmH2O). The channels they come from are
`recording.pressure_channel` and `recording.tidal_volume_channel`.

## 6. Limits

- Tested on four decremental trials from two patients (volume control with an
  end-inspiratory pause, Dräger PressurePod); other ventilation modes and pressure
  sources on synthetic data only.

*Research use only - not a medical device.*
