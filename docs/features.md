# fastEIT - Clinical Indices (layer 3)

Layer 3 turns a preprocessed recording (`EITRecording`, [`preprocessing.md`](preprocessing.md))
or a segmented PEEP trial (`PeepTrial`, [`segmentation.md`](segmentation.md)) into tables of
clinical indices: one row per breath, per breath and region, per PEEP step, per step and
region, and a summary. The tables are ready for a database or for R: the clinical context
(patient, position, side in park bench) is joined at the time of the analysis, the
recording does not need to know it.

The definitions and the evidence are in `notebooks/07_clinical_indices.ipynb` (11 tidal
recordings, four PEEP trials); on the four trials the functions below give the numbers of
the notebook.

1. [Entry points](#1-entry-points)
2. [Indices of a breath](#2-indices-of-a-breath)
3. [Tidal recordings](#3-tidal-recordings)
4. [PEEP trials](#4-peep-trials)
5. [Limits](#5-limits)

---

## 1. Entry points

```python
from fasteit.features import peep_trial_features, tidal_features
from fasteit.models import RecordingContext, Segment

# A recording at one setting (or the tidal phase of a longer one).
recording = preprocess(load_data("patient01.bin"),
                       context=RecordingContext(tidal_volume_ml=450))  # Vt optional
tidal = tidal_features(recording)            # or phase=Segment(0, 300, category="phase", label="tidal")
tidal.breaths    # one row per breath
tidal.regions    # one row per breath and region
tidal.summary    # median and quartiles over the valid breaths, by region and index

# A PEEP trial.
trial = segment_peep_trial(preprocess(load_data("peep_trial.bin"), context=context))
features = peep_trial_features(trial)
features.steps          # one row per PEEP step
features.step_regions   # one row per step and region
features.summary        # crossing PEEP, best PEEP
```

## 2. Indices of a breath

All from the frames of the layer 2 breath (one sample, no averaging): start =
end-expiratory valley, peak = end-inspiratory peak.

| Column | Definition | Source |
|--------|------------|--------|
| `eeli` | signal inside the lung mask at the start of the breath (arbitrary units) | TREND, Frerichs 2017, DOI: 10.1136/thoraxjnl-2016-208357 |
| `tiv` | signal at the peak minus at the start | TREND |
| `gi` | global inhomogeneity: sum of \|pixel TIV - median\| / sum of pixel TIV, over the lung pixels | Zhao 2009, DOI: 10.1007/s00134-009-1589-y |
| `cov_vd`, `cov_rl` | centre of ventilation on the image grid: 0 = ventral / patient right, 1 = dorsal / patient left | Frerichs 2006, DOI: 10.1164/rccm.200512-1942OC |
| `duration`, `inspiratory_time`, `expiratory_time`, `ti_ttot`, `respiratory_rate` | from the EIT breath (impedance), not from the ventilator | - |
| `peep`, `plateau`, `driving_pressure` | from layer 2 (cmH2O) | notebook 06 |
| `tidal_volume`, `tidal_volume_source` | typed for the step, typed for the recording, or measured (mL) | layer 2 and 2b |
| `compliance` | TIV / driving pressure (EIT units per cmH2O) | - |
| `compliance_ml` | tidal volume / driving pressure (mL/cmH2O) | - |

By region (`regions`, one row per breath and region; regions of layer 2: ventral and
dorsal halves, layers, right and left, quadrants):

| Column | Definition |
|--------|------------|
| `eeli`, `tiv` | the same on the signal of the region |
| `share` | TIV of the region / TIV of the lung (0-1): a ratio of sums |
| `tiv_ml` | share × tidal volume (mL) |
| `compliance` | TIV of the region / driving pressure (EIT units per cmH2O) |
| `compliance_ml` | share × tidal volume / driving pressure (mL/cmH2O) |

The two compliances are both kept: the one in mL/cmH2O is that of the respiratory system,
as the ventilator would show it; the one in EIT units also follows what changes in the
imaged slice (in volume control the same tidal volume gives 80-105 % of the TIV of the
first step across a PEEP trial) and moves the best PEEP and the crossing point.

## 3. Tidal recordings

`tidal_features(recording, phase=None)` gives the indices of every breath, and a summary
over the **valid breaths** (no quality flag, not excluded) inside the tidal phase: one
row per region (`"global"` for the whole lung) and index (`feature`), with `median`, `q1`,
`q3` and `n_breaths` (breaths with a value). The tidal volume of the breaths is the one
typed in `RecordingContext.tidal_volume_ml`, otherwise the one measured on the ventilator
curves.

## 4. PEEP trials

`peep_trial_features(trial)` gives the indices of every breath (with `peep_step` and
`analysed`) and, for each step, from its analysed breaths:

| Column | Definition |
|--------|------------|
| `respiratory_rate`, `eeli`, `tiv`, `gi`, `cov_vd`, `cov_rl` | median of the analysed breaths |
| `driving_pressure`, `tidal_volume` (and their sources) | from the step table of layer 2b |
| `compliance` | median TIV / driving pressure of the step: it also works when the driving pressure comes from one inspiratory hold or is typed |
| `compliance_ml` | tidal volume / driving pressure of the step |
| `compliance_pct_of_best`, `compliance_ml_pct_of_best` | as % of the best step |
| `collapse`, `overdistension` | Costa method (%), from the median tidal image of the step: pixel compliance, loss as % of the best of each pixel, weighted by the best compliance (Costa 2009, DOI: 10.1007/s00134-009-1447-y; formulas as in Shono and Kotani 2019, DOI: 10.1186/s40560-019-0358-4) |
| `d_eeli_tiv_vs_previous`, `d_eeli_tiv_vs_first` | change of EELI in tidal volumes: change / TIV of the reference step |
| `d_eeli_ml_vs_previous`, `d_eeli_ml_vs_first` | the same in mL: × tidal volume of the reference step |
| `d_compliance_pct_of_best_vs_...`, `d_compliance_ml_pct_of_best_vs_...`, `d_collapse_vs_...`, `d_overdistension_vs_...` | changes vs the previous and the first step, in points |

`step_regions` has, for each step and region, the medians of `eeli`, `tiv` and `share` of
the analysed breaths, and `tiv_ml`, `compliance` and `compliance_ml` of the step. The
summary gives the crossing PEEP of collapse and overdistension (linear interpolation; the
lowest if the curves cross more than once) and the PEEP of the best global compliance in
EIT units and in mL/cmH2O.

## 5. Limits

- Tested on four decremental PEEP trials from two patients (volume control, tidal volume as
  set on the ventilator) and on synthetic recordings with a known answer; the Costa method
  is implemented from its published formulas, not checked against another implementation.
- EELI is in arbitrary units: only its changes are comparable, and in mL only through the
  tidal volume. The change of EELI follows the change of end-expiratory lung volume
  linearly during PEEP changes (Hinz et al., Intensive Care Med 2003, DOI:
  10.1007/s00134-002-1555-4); the conversion to mL further assumes the same impedance per
  mL for the tidal and the end-expiratory change, in every region and at every PEEP. In the
  four trials it gives 1.6-2.3 times compliance × change of PEEP: an estimate for research
  on groups, not a measurement of lung volume.
- The mL of a region are the tidal volume shared as in the imaged slice: the regions sum
  to the tidal volume, so they are a distribution, not independent measurements; not
  corrected for the uneven sensitivity of EIT, a different impedance per mL in diseased
  regions, or the lung outside the slice.
- Compliance from the airway pressure is that of the respiratory system (lung and chest
  wall), valid in passive patients; regional compliance uses the same driving pressure for
  every region. The limits of the pressures and of the tidal volume are in
  [`preprocessing.md`](preprocessing.md), section 6.
- Which regions are dependent depends on the position (ventral/dorsal supine, right/left
  on the side): the tables have every region, the choice belongs to the analysis.
- One lung mask for the whole recording.

*Research use only - not a medical device.*
