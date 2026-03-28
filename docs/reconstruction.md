# ML Reconstruction Pipeline

Data-driven EIT image reconstruction using Ridge regression.
Learns the Dräger PulmoVista 500 reconstruction mapping from paired
`.eit` / `.bin` recordings.

**Requires**: `pip install fasteit[reconstruction]` (adds `scikit-learn`).

**Device-specific**: trained and validated on Dräger PulmoVista 500 only
(16 electrodes, adjacent drive, 208 measurements per frame, 32x32 pixel output).

## Overview

The Dräger PulmoVista reconstructs 32x32 impedance images from 208 raw
transimpedance measurements using a proprietary Newton-Raphson FEM algorithm.
This module learns that mapping directly from data using Ridge regression
(L2-regularised linear regression), which is physically motivated: the EIT
forward model is linear, so the inverse map is approximately linear.

Reference:
> Hoerl AE, Kennard RW. "Ridge Regression: Biased Estimation for
> Nonorthogonal Problems." *Technometrics* 12(1), 1970, pp. 55-67.
> DOI: 10.1080/00401706.1970.10488634

## Modules

### `data_prep` -- data loading and normalisation

```python
from fasteit.reconstruction import load_paired, normalize, prepare_dataset
```

| Function | Description |
|----------|-------------|
| `load_paired(eit_path, bin_path, input_mode)` | Load paired .eit/.bin, return `(X, Y)` arrays. `input_mode="vv"` gives 208 calibrated features; `"raw"` gives 416 raw `[trans_A, trans_B]` features. |
| `normalize(arr, n_ref=50)` | Baseline subtraction: subtract mean of first `n_ref` frames. Returns `(delta, ref)`. |
| `prepare_dataset(pairs, n_ref, input_mode)` | Load multiple pairs, normalise per-file, concatenate. |

**Normalisation rationale**: absolute impedance varies across patients and after
device recalibration. Per-file baseline subtraction removes this offset, leaving
only differential signals (tidal variations, EELI trends).

### `ridge_model` -- RidgeReconstructor

```python
from fasteit.reconstruction import RidgeReconstructor

model = RidgeReconstructor(alpha=100.0)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
r2 = model.score(X_test, Y_test)

model.save("model.npz")
loaded = RidgeReconstructor.load("model.npz")
```

| Method | Description |
|--------|-------------|
| `fit(X, Y)` | Fit Ridge regression on normalised, scaled data. |
| `predict(X)` | Predict pixel values, shape `(N, 1024)`. |
| `score(X, Y)` | R² coefficient of determination (sklearn convention). |
| `save(path)` | Save weights to `.npz` (no pickle -- portable, auditable). |
| `load(path)` | Class method to reconstruct a fitted model from `.npz`. |

**Important**: `RidgeReconstructor` does not include a `StandardScaler`.
The caller must fit a scaler on training data and apply it before calling
`fit()` and `predict()`. The scaler must be saved separately for inference.

### `metrics` -- evaluation metrics

```python
from fasteit.reconstruction import summary_metrics

m = summary_metrics(Y_true, Y_pred)
# {'r2': 0.97, 'rmse': 0.60, 'mse_mean': ..., 'corr_spatial_mean': ..., 'corr_global': ...}
```

| Function | Returns | Clinical meaning |
|----------|---------|------------------|
| `mse_per_frame(Y_true, Y_pred)` | `(N,)` | Per-frame reconstruction error |
| `rmse_global(Y_true, Y_pred)` | scalar | Overall error in pixel units |
| `r2_score(Y_true, Y_pred)` | scalar | Fraction of variance explained (pooled) |
| `error_map(Y_true, Y_pred)` | `(32, 32)` | Spatial MAE per pixel |
| `correlation_per_frame(Y_true, Y_pred)` | `(N,)` | Spatial pattern fidelity per frame |
| `global_signal_correlation(Y_true, Y_pred)` | scalar | Tidal waveform fidelity over time |
| `summary_metrics(Y_true, Y_pred)` | dict | All metrics in one call |

**Note on `r2_score`**: computes a pooled R² over all elements (frames x pixels),
not per-pixel R² averaged. See docstring for details on the difference from
sklearn's `r2_score(multioutput='uniform_average')`.

## Input modes

| Model | Input | Features | Description |
|-------|-------|----------|-------------|
| v1 | `"vv"` | 208 | Calibrated transimpedances (Adler formula with gain x I_injection) |
| v1b | `"raw"` | 416 | Raw `[trans_A, trans_B]` -- bypasses hardware calibration constants |

v1b is more robust at recording boundaries because it does not depend on
hardware-specific calibration constants.

## Current results (proof of concept)

- 2 patients, 10 recordings, ~89k frames
- 80/20 sequential split per recording, per-file baseline normalisation
- Alpha selected via validation split (no test-set leakage)

| Model | R² test | Spatial corr | Global corr |
|-------|---------|-------------|-------------|
| v1 (208) | 0.950 | 0.985 | 0.944 |
| **v1b (416)** | **0.951** | **0.986** | **0.945** |

After outlier filtering (1 glitch frame removed):
R² = 0.972, spatial corr = 0.987, global corr = 0.986.

## Known limitations

- **2 patients only** -- cross-patient generalisation not yet proven. Requires
  more patients for Leave-One-Patient-Out CV (LOPO-CV).
- **StandardScaler not saved with model** -- must be persisted separately.
  Will be integrated into a Pipeline in the planned `EITReconstructor` class.
- **No sentinel handling** -- bad frames (sentinel values from parser) are not
  filtered before training. A single corrupted frame can bias the weight matrix.
- **Dräger-only** -- cross-vendor (Infivision) not yet tested.

## Notebook

See [`notebooks/04_ml_reconstruction.ipynb`](../notebooks/04_ml_reconstruction.ipynb)
for the full training and evaluation pipeline.
