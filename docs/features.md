# fastEIT - Clinical Indices (layer 3)

Layer 3 turns a preprocessed recording (`EITRecording`, [`preprocessing.md`](preprocessing.md))
or a segmented PEEP trial (`PeepTrial`, [`segmentation.md`](segmentation.md)) into tables of
clinical indices: one row per breath, per breath and region, per PEEP step, per step and
region, and a summary. The tables are ready for a database or for R: the clinical context
(patient, position, side in park bench) is joined at the time of the analysis, the
recording does not need to know it.

The definitions and the evidence are in `notebooks/07_clinical_indices.ipynb` (11 tidal
recordings, four PEEP trials); on the four trials the functions below give the crossing
PEEP, the best PEEP and the change of EELI in mL of the notebook.

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
features.summary        # crossing PEEP, best PEEP, R/I (global and by region)
```

The object holds everything: `features.trial.steps` (driving pressure and tidal volume of
each step, breaths analysed), `features.trial.breaths` (step of each breath),
`features.trial.recording.breaths` (times, quality flags, PEEP, plateau, driving
pressure, measured tidal volume). Each value is stored once: the tables of layer 3 have
the keys (`breath_id`, `step`, `peep`, `region`) and what layer 3 computes, and are
joined on the keys with the tables before them. The flat tables with everything in one
row per breath and per step, for R, SPSS and databases, are made by layer 4
([`export.md`](export.md)).

## 2. Indices of a breath

All from the frames of the layer 2 breath (one sample, no averaging): start =
end-expiratory valley, peak = end-inspiratory peak.

| Column | Definition | Source |
|--------|------------|--------|
| `eeli` | signal inside the lung mask at the start of the breath (arbitrary units) | TREND, Frerichs 2017, DOI: 10.1136/thoraxjnl-2016-208357 |
| `tiv` | signal at the peak minus at the start | TREND |
| `gi` | global inhomogeneity: sum of \|pixel TIV - median\| / sum of pixel TIV, over the lung pixels | Zhao 2009, DOI: 10.1007/s00134-009-1589-y |
| `cov_vd`, `cov_rl` | centre of ventilation on the image grid: 0 = ventral / patient right, 1 = dorsal / patient left | Frerichs 2006, DOI: 10.1164/rccm.200512-1942OC |
| `respiratory_rate` | 60 / duration of the EIT breath (breaths/min) | - |
| `tidal_volume_used`, `tidal_volume_source` | tidal volume used for the values in mL: typed for the step, typed for the recording, or measured | layer 2 and 2b |
| `compliance` | TIV / driving pressure (EIT units per cmH2O) | - |
| `compliance_ml` | tidal volume / driving pressure (mL/cmH2O): static compliance of the respiratory system, with the tidal volume typed or measured | - |

By region (`regions`, one row per breath and region; regions of layer 2: ventral and
dorsal halves, layers, right and left, quadrants):

| Column | Definition |
|--------|------------|
| `eeli`, `tiv` | the same on the signal of the region |
| `share` | TIV of the region / TIV of the lung (0-1): a ratio of sums |
| `tiv_ml` | share × tidal volume (mL) |
| `compliance` | TIV of the region / driving pressure (EIT units per cmH2O): the sum of the compliances of its pixels, which are in parallel under the same driving pressure |
| `compliance_ml` | share × tidal volume / driving pressure (mL/cmH2O) |

The two compliances are both kept: the one in mL/cmH2O is the static compliance of the
respiratory system; the one in EIT units also follows what changes in the imaged slice
(in volume control the same tidal volume gives 80-105 % of the TIV of the first step
across a PEEP trial) and can move the best PEEP (B park bench: 4.5 in EIT units, 8.5 in
mL/cmH2O). Costa is computed in EIT units only. A region's compliance is a partial
compliance, its share of the total: a small region has a low compliance even when it is
healthy.

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
| `compliance` | median TIV / driving pressure of the step: it also works when the driving pressure comes from one inspiratory hold or is typed |
| `compliance_ml` | tidal volume / driving pressure of the step (it needs no EIT data, so it exists also for a step without analysed breaths) |
| `compliance_pct_of_best`, `compliance_ml_pct_of_best` | as % of the best step |
| `collapse`, `overdistension` | Costa method (%), from the median tidal image of the step: pixel compliance, loss as % of the best of each pixel, weighted by the best compliance (Costa 2009, DOI: 10.1007/s00134-009-1447-y; formulas as in Shono and Kotani 2019, DOI: 10.1186/s40560-019-0358-4) |
| `d_eeli_tiv_vs_previous`, `d_eeli_tiv_vs_first` | change of EELI in tidal volumes: change / TIV of the step at the lower PEEP of the two (Wang 2020, DOI: 10.12659/MSM.922609) |
| `d_eeli_ml_vs_previous`, `d_eeli_ml_vs_first` | the same in mL: × tidal volume of that step |
| `d_compliance_pct_of_best_vs_...`, `d_compliance_ml_pct_of_best_vs_...` | changes of compliance vs the previous and the first step, in points of % of the best |

Collapse and overdistension are read on their curves (one value per step); a change between
two PEEP levels is the difference of two values of the curve.

`step_regions` has, for each step and region, the medians of `eeli`, `tiv` and `share` of
the analysed breaths, and `tiv_ml`, `compliance` and `compliance_ml` of the step.

The summary is a long table (`region`, `feature`, `value`):

| Feature | Region | Definition |
|---------|--------|------------|
| `crossing_peep` | global | PEEP where collapse and overdistension cross (linear interpolation; the lowest if they cross more than once) |
| `best_peep_compliance`, `best_peep_compliance_ml` | global | PEEP of the best global compliance, in EIT units and in mL/cmH2O |
| `ri_peep_high`, `ri_peep_low` | global | PEEP of the two steps used for R/I |
| `ri` | global and each region | recruitment-to-inflation ratio |
| `recruited_volume_ml` | global and each region | recruited volume (mL, with a tidal volume) |

**Recruitment-to-inflation ratio** (Chen et al., Am J Respir Crit Care Med 2020, DOI:
10.1164/rccm.201902-0334OC), between the step nearest to PEEP 15 and the step nearest to
PEEP 5 at least 10 below it (`AnalysisConfig`):

```
dEELV = (EELI_high - EELI_low) x Vt_low / TIV_low        (Wang 2020)
recruited volume = dEELV - C_low x dPEEP                 C_low = Vt_low / driving_low
R/I = (recruited volume / dPEEP) / C_low = (EELI_high - EELI_low) / (C_EIT,low x dPEEP) - 1
```

The tidal volume cancels in R/I, which needs only EIT and the driving pressure; a region
uses its own EELI and compliance. The original method reads one expiration after the drop
of PEEP; here each level is the equilibrium EELI of a step, as in Wang 2020.

Steps with fewer analysed breaths than the minimum of layer 2b
(`enough` false) are kept, with a warning: filter them in the analysis if needed. If the
first step has no analysed breaths, the changes vs the first step are NaN.

## 5. Limits

- Tested on four decremental PEEP trials from two patients (volume control, tidal volume as
  set) and on synthetic data with a known answer; Costa implemented from its published
  formulas, not checked against another implementation.
- EELI is in arbitrary units: only its changes are comparable. The change in mL assumes the
  same impedance per mL for the tidal and the end-expiratory change (Hinz et al.,
  Intensive Care Med 2003, DOI: 10.1007/s00134-002-1555-4, show the linear relation of the
  changes); in the four trials it is 1.6-2.3 times compliance × change of PEEP, cause
  unknown: exploratory, not a measurement of lung volume.
- The mL of a region are the tidal volume shared as in the imaged slice (they sum to the
  tidal volume): a distribution, not independent measurements.
- Compliance from the airway pressure is that of the respiratory system, in passive
  patients; regional compliance is a partial compliance with the same driving pressure
  for every region.
- % of the best step, Costa and the changes vs the first step depend on the PEEP range
  tested: patients are comparable with the same range and steps.
- Costa: an error common to a step (its driving pressure, the share of the tidal volume
  seen by the slice) appears as collapse or overdistension.
- R/I mixes recruitment and overdistension (Zhao et al., Crit Care 2023, DOI:
  10.1186/s13054-023-04520-8); the recruited volume from EIT agreed with the flow-derived
  one with limits of -182 to +144 mL in pigs (Wang 2020). A region that is almost
  collapsed at the low PEEP has a small compliance there, and its R/I becomes large: read
  it with its recruited volume.
- One lung mask for the whole recording (in the four trials the pixels ventilated at some
  step but outside it carry at most 1-3 % of the TIV); GI not comparable with cut-offs
  computed on other masks.

*Research use only - not a medical device.*
