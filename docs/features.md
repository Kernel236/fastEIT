# fastEIT - Clinical Indices (layer 3)

Layer 3 turns a preprocessed recording (`EITRecording`, [`preprocessing.md`](preprocessing.md))
or a segmented PEEP trial (`PeepTrial`, [`segmentation.md`](segmentation.md)) into tables of
clinical indices: one row per breath, per breath and region, per PEEP step, per step and
region, and a summary. The tables are ready for a database or for statistical analysis: the clinical context
(patient, position, side in park bench) is joined at the time of the analysis, the
recording does not need to know it.

The definitions and the evidence are in `notebooks/07_clinical_indices.ipynb` (11 tidal
recordings, four PEEP trials); on the four trials the functions below give the crossing
PEEP, the best PEEP and the change of EELI in mL of the notebook.

Steps for a user: [analyse a recording](guide/analyse.md).

1. [Entry points](#1-entry-points)
2. [Indices of a breath](#2-indices-of-a-breath)
3. [Tidal recordings](#3-tidal-recordings)
4. [PEEP trials](#4-peep-trials)
5. [Limits](#5-limits)

---

## 1. Entry points

```python
from fasteit.config import AnalysisConfig, SegmentationConfig
from fasteit.features import peep_trial_features, tidal_features
from fasteit.models import RecordingContext, Segment
from fasteit.parsers.loader import load_data
from fasteit.preprocessing import preprocess
from fasteit.segmentation import segment_peep_trial

# A recording at one setting (or the tidal phase of a longer one).
recording = preprocess(load_data("patient01.bin"),
                       context=RecordingContext(tidal_volume_ml=450))  # Vt optional
tidal = tidal_features(recording)            # or phase=Segment(0, 300, category="phase", label="tidal")
tidal.breaths    # one row per breath
tidal.regions    # one row per breath and region
tidal.summary    # median and quartiles over the valid breaths, by region and index

# A PEEP trial, with the PEEP levels of its protocol and the two levels of R/I.
context = RecordingContext(protocol="peep_trial", tidal_volume_ml=450)
protocol = SegmentationConfig(peep_levels=(16, 14, 12, 10, 8, 6, 4, 2, 0))
trial = segment_peep_trial(preprocess(load_data("peep_trial.bin"), context=context),
                           protocol)
features = peep_trial_features(trial, AnalysisConfig(ri_peep_high=14, ri_peep_low=4))
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
row per breath and per step, for statistical analysis and databases, are made by layer 4
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
| `driving_pressure_used`, `driving_pressure_source` | driving pressure used for the compliance: typed for the step (it wins, as in the step table), or read on the curve for the breath | layer 2 and 2b |
| `compliance` | TIV / driving pressure (EIT units per cmH2O) | - |
| `compliance_ml` | tidal volume / driving pressure (mL/cmH2O): static compliance of the respiratory system, with the tidal volume and the driving pressure typed or measured | - |

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
over the **valid breaths** (no quality flag, not excluded) inside the tidal phase (a
breath is inside when it starts and ends inside it): one
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
| `collapse`, `overdistension` | Costa method (%), from the median tidal image of the step: pixel compliance, loss as % of the best of each pixel, weighted by the best compliance; the loss of a pixel is at most 100 % (Costa 2009, DOI: 10.1007/s00134-009-1447-y; formulas as in Shono and Kotani 2019, DOI: 10.1186/s40560-019-0358-4) |
| `d_eeli_tiv_vs_previous`, `d_eeli_tiv_vs_first` | change of EELI in tidal volumes: change / TIV of the step at the lower PEEP of the two (Wang 2020, DOI: 10.12659/MSM.922609) |
| `d_eeli_ml_vs_previous`, `d_eeli_ml_vs_first` | the same in mL: × tidal volume of that step |
| `d_compliance_pct_of_best_vs_...`, `d_compliance_ml_pct_of_best_vs_...` | changes of compliance vs the previous and the first step, in points of % of the best |

Collapse and overdistension are read on their curves (one value per step); a change between
two PEEP levels is the difference of two values of the curve.

`step_regions` has, for each step and region, the medians of `eeli`, `tiv` and `share` of
the analysed breaths, `tiv_ml`, `compliance` and `compliance_ml` of the step, and Costa
`collapse` and `overdistension` of the region.

**Costa by region.** The same formula on the pixels of the region only, its denominator
included, so the value is a % of the region ("30 % of the dorsal lung collapsed"); the
best of each pixel is its own, over all the steps:

```
collapse_region(step) = sum over the region's pixels below their best PEEP of (loss x best)
                        / sum over the region's pixels of best
costa_weight_region   = sum of best over the region's pixels / sum of best over the lung
collapse_region x costa_weight_region = the region's part of the collapse of the lung
```

The regions of a partition (ventral and dorsal, right and left) therefore do not add up
to the value of the lung: their mean weighted by `costa_weight` does. Costa has been
reported by right and left lung (Mlcek et al., BMC Pulm Med 2021, DOI:
10.1186/s12890-021-01501-x, as % of the whole lung: the value of the region x its weight)
and by ventral and dorsal half (Jonkman et al., Am J Respir Crit Care Med 2023, DOI:
10.1164/rccm.202212-2300OC, halves of the image; here the halves are cut on the lung,
see the [regions](preprocessing.md)). In a lateral position the gravitational axis is
right-left: the dependent lung is the side the patient lies on (`right` and `left` are
the patient's).

The summary is a long table (`region`, `feature`, `value`):

| Feature | Region | Definition |
|---------|--------|------------|
| `crossing_peep` | global | PEEP where collapse and overdistension cross (linear interpolation; the lowest if they cross more than once) |
| `best_peep_compliance`, `best_peep_compliance_ml` | global | PEEP of the best global compliance, in EIT units and in mL/cmH2O |
| `ri_peep_high`, `ri_peep_low` | global | levels of the protocol of the two steps used for R/I |
| `ri` | global and each region | recruitment-to-inflation ratio |
| `recruited_volume_ml` | global and each region | recruited volume (mL, with a tidal volume) |
| `costa_weight` | each region | best compliance of the region / that of the lung (0-1), over the pixels Costa uses |

`crossing_peep` and the best PEEP are on the levels of the protocol
(`peep_set`, from `SegmentationConfig.peep_levels`) when every step has one, so that they
can be compared between patients; otherwise on the PEEP of the steps (`peep`: measured
for the steps found from the pressure, typed for the steps placed by hand).

**Recruitment-to-inflation ratio** (Chen et al., Am J Respir Crit Care Med 2020, DOI:
10.1164/rccm.201902-0334OC), between the steps at the two PEEP levels of the protocol
given in `AnalysisConfig` (`ri_peep_high`, `ri_peep_low`): 15 and 5 in the original
method, for example 14 and 4 in a trial with steps of 2 from 16. There is no R/I without
the two levels, or when the trial does not have exactly one step at each of them (a
warning). The steps are never chosen by nearness: with steps of 2, 16 and 14 are equally
near to 15, and a measured drop of 9.99 instead of 10 skipped the step at 4, so the pair
changed between patients of the same study. The drop of PEEP in the formula is computed
on `peep`: the change of pressure that happened, or the levels typed when the pressure
curve is not reliable.

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

- Costa is implemented from its published formulas and checked on synthetic data with a
  known answer, not against another implementation.
- EELI is in arbitrary units: only its changes count. Its change in mL assumes the same
  impedance per mL for the tidal and the end-expiratory change (Hinz et al., Intensive
  Care Med 2003, DOI: 10.1007/s00134-002-1555-4); where it was checked it was 1.6-2.3
  times compliance × change of PEEP, cause unknown, so it is exploratory.
- The mL of a region share the tidal volume as seen in the imaged slice: a distribution,
  not independent measurements.
- Compliance is that of the respiratory system, in a passive patient; regional compliance
  uses the same driving pressure for every region.
- % of the best step, Costa and the changes vs the first step depend on the PEEP range
  tested: compare patients with the same steps.
- Costa reads an error common to a step (its driving pressure, the share of the tidal
  volume seen by the slice) as collapse or overdistension.
- Costa by region is a % of the region: with a small `costa_weight` (a small or little
  ventilated region) a few pixels move it a lot. Pixels never ventilated in the trial are
  outside the lung mask and count nowhere, globally or by region.
- R/I mixes recruitment and overdistension (Zhao et al., Crit Care 2023, DOI:
  10.1186/s13054-023-04520-8) and becomes large in a region almost collapsed at the low
  PEEP.
- One lung mask for the whole recording: GI is not comparable with cut-offs computed on
  other masks.

*Research use only - not a medical device.*
