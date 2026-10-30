# fastEIT - Segmentation Layer (PEEP trials)

The segmentation layer (layer 2b) takes a preprocessed recording (`EITRecording`) and
splits it by protocol. For a PEEP trial it finds the PEEP steps, reads the airway
pressures of each breath and chooses the breaths analysed in each step. It is still
preparation of the data: the clinical indices (EELI, TIV, compliance, Costa) come in
layer 3.

The evidence for the pressure readings and the step detection is in
`notebooks/06_peep_trial_segmentation.ipynb` (four decremental PEEP trials from two
patients). The 15 s left out after each change, the 10 (at least 5) breaths analysed and
the 0.5 mbar tolerance are protocol choices.

1. [Entry point](#1-entry-point)
2. [Steps](#2-steps)
3. [Recordings with more than one protocol](#3-recordings-with-more-than-one-protocol)
4. [Output](#4-output)
5. [Limits](#5-limits)

---

## 1. Entry point

```python
from fasteit.models import Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

recording = preprocess(load_data("peep_trial.bin"))
trial = segment_peep_trial(recording)

trial.steps              # one row per PEEP step: level, breaths, breaths analysed
trial.analysed_breaths   # the breaths to analyse, with PEEP, plateau, driving pressure
print("\n".join(trial.log))
```

Without a pressure signal the steps are placed by hand, one segment per step with its
PEEP level:

```python
steps = [
    Segment(60, 100, category="peep_step", parameters={"peep": 16}),
    Segment(110, 150, category="peep_step", parameters={"peep": 14}),
]
trial = segment_peep_trial(recording, peep_steps=steps)
```

Steps placed by hand replace the automatic ones; the breaths analysed are then the
valid breaths inside each segment (a breath cut by the edge of a segment is left out).
Without a pressure signal the plateau read on the ventilator can be typed with the step,
`parameters={"peep": 16, "plateau": 28}`, or directly the driving pressure,
`parameters={"peep": 16, "driving_pressure": 12}` (e.g. in pressure control, where it is
set). When there is a
pressure signal, a typed PEEP more than 1 mbar from the measured one gives a warning.
Parameters are in `SegmentationConfig`.

## 2. Steps

| # | Step | Module |
|---|------|--------|
| 1 | Airway pressure: the first available of PressurePod, Medibus or Timpel `airway_pressure`; without a waveform, the PEEP and plateau computed by the ventilator (Medibus) | `pipeline` |
| 2 | Pressures of each breath, all from the same channel: start of inspiration on the pressure curve, PEEP = median pressure in the 0.2 s before it, plateau = median pressure at the end of an end-inspiratory hold of at least 0.5 s, driving pressure = plateau - PEEP | `breath_pressures` |
| 3 | Steps: a new step when 3 breaths in a row are more than 1 mbar (half of the step of the protocol) from the current level and within 1 mbar of each other; level = median PEEP. Excluded breaths (maneuvers) do not count | `peep_steps` |
| 4 | Breaths analysed: the first 15 s after each change are left out; then the last valid breaths whose PEEP is within 0.5 mbar of the step level, up to 10; fewer than 5 give a warning | `peep_steps` |

The driving pressure of each step (the input of compliance in layer 3) works with the
usual protocols:

| Protocol | Driving pressure of the step (`driving_pressure_source`) |
|----------|-------------------------------------------------------|
| Volume control with a continuous end-inspiratory pause | median of the analysed breaths (`"breaths"`) |
| Volume control with one inspiratory hold per step | median of the breaths with a hold, one is enough (`"hold"`) |
| Pressure control | the flat inspiratory pressure is read as plateau (`"breaths"`), valid if the flow is zero at the end of inspiration |
| No pressure signal | typed with the step placed by hand (`"typed"`) |

The event texts typed on the device are not used to cut the steps: in the trials of the
notebook they were written from 2 s after to 14 s before the change of pressure.

## 3. Recordings with more than one protocol

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

## 4. Output

`PeepTrial` fields:

| Field | Content |
|-------|---------|
| `recording`, `config` | the preprocessed recording and the parameters used |
| `pressure_channel`, `pressure_unit` | column of `recording.aux` the pressures were read from (None if none), and its unit |
| `source` | `"pressure"` or `"manual"` |
| `steps` | `step`, `peep` (level), `start_time`, `end_time`, `n_breaths`, `n_analysed`, `enough`, `plateau`, `driving_pressure`, `driving_pressure_source`, `n_driving_pressure`, `enough_driving_pressure` |
| `breaths` | the breath table of the recording plus `peep`, `plateau`, `driving_pressure`, `peep_step`, `analysed` |
| `log` | what the segmentation did, with warnings |

Pressures are in the units of the recording: mbar for Dräger (1 mbar = 1.02 cmH2O),
cmH2O for Timpel.

## 5. Limits

- Tested on four decremental trials from two patients, in volume control with an
  end-inspiratory pause, with the Dräger PressurePod; Timpel and Medibus pressures are
  tested on synthetic data only. Medibus values are updated once per ventilator breath:
  at a change of step they can lag by one breath.
- The PEEP is the end-expiratory pressure at the airway opening: it does not include
  intrinsic PEEP, so it is a lower bound of the total PEEP (in one trial it read 0.2-0.5
  mbar above the set value, 1 mbar at PEEP 0, at 19 breaths/min).
- The plateau needs at least 0.5 s of hold (volume control) or of flat inspiratory
  pressure (pressure control, valid only if the flow is zero at the end of
  inspiration); otherwise it is missing, and the peak pressure is never used instead.
  Inspiratory holds and pressure control are tested on synthetic data only. The plateau and its detection were not validated against the
  ventilator plateau or a manual hold.
- The driving pressure is that of the respiratory system: the oesophageal and
  transpulmonary pressures of the PressurePod are not used. Compliance from the airway
  pressure is valid only in passive patients, and without a tidal volume it is in EIT
  units per mbar.
- Short steps at a low rate leave few breaths after the first 15 s (4-9 in the trials at
  11-13 breaths/min with 40 s steps): the warning reports it.

*Research use only - not a medical device.*
