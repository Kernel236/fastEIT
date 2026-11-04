# ML Reconstruction Pipeline

Data-driven EIT image reconstruction with ridge regression: it learns the Dräger PulmoVista 500
reconstruction from paired `.eit` / `.bin` recordings.

**Requires**: `pip install fasteit[reconstruction]` (adds `scikit-learn`).

**Device-specific**: trained and validated on Dräger PulmoVista 500 only (16 electrodes, adjacent
drive, 208 measurements per frame, 32x32 pixel output). For research only, not a medical device.
Trained models are not distributed: only the method and its results are published.

## The idea

The device stores, for every frame, the 208 transimpedances (`.eit`) and the image it reconstructs
from them (`.bin`). If the reconstruction is (close to) a fixed linear map, a matrix that turns the
208 measurements into the 1024 pixels can be learned from the pairs: a supervised learning problem,
with the device's own images as targets.

Reference:
> Hoerl AE, Kennard RW. "Ridge Regression: Biased Estimation for
> Nonorthogonal Problems." *Technometrics* 12(1), 1970, pp. 55-67.
> DOI: 10.1080/00401706.1970.10488634

## Method

Each recording is normalised to its own reference, taken at its bottom: the 2% of its frames where
the median relative change of the 208 channels is lowest (the lowest end-expirations, found from X
alone). This is where the Dräger images have their zero. X and Y take their reference at those frames:

- **X** (208 calibrated transimpedances): relative change `(x - ref) / ref`. Contact, thorax size
  and gain multiply a whole channel; the division cancels that factor. A channel whose reference is
  below `BAD_REF_FRAC` = 0.2 x its typical value, or has the unusual sign, is set to 0 (a bad
  electrode). Typical value and usual sign come from the training recordings only.
- **Y** (1024 pixels): `y - ref`, zero at the lowest end-expiration. Clinical indices are changes
  (TIV, dEELI) or ratios of changes, so the offset does not matter.
- **Low-pass**: the reconstruction of a whole recording is filtered at 50/min with zero phase, as
  the Dräger images are (`lowpass`).
- **Model**: X and Y standardised on the training data, ridge, prediction back to pixel units.

Evaluation is leave-one-patient-out. A filter that uses the `.bin` (recordings whose normalised pixels have
an RMS more than 10 times the median, `pixel_rms`: Dräger images far beyond breathing, which would
drive the least squares) selects the training data only; every patient
is evaluated on all their recordings. Besides the R², the notebook reports the R² of the changes:
each pixel minus its mean in each recording, for both the `.bin` and the prediction.

## API

```python
from fasteit.parsers.loader import load_data
from fasteit.reconstruction import EITReconstructionModel, fit_model, load_paired, reconstruct

# training: raw paired recordings, one (x, y) each
model = fit_model([load_paired(eit, bin_) for eit, bin_ in training_pairs])
model.save("model.npz")                              # stays local, never in the repository

# reconstruction of a new .eit, ready for preprocess()
model = EITReconstructionModel.load("model.npz")
images = reconstruct(load_data("rec.eit"), model)    # low-pass 50/min; lowpass_bpm=None: off
```

| Function | Description |
|----------|-------------|
| `load_paired(eit_path, bin_path, input_mode)` | Paired `(X, Y)`; `"vv"`: 208 calibrated features, `"raw"`: 416 `[trans_A, trans_B]`. |
| `reference_frames(x, low, min_frames)` | Lowest end-expirations of a whole recording: the lowest `low` (2%) of the median relative change, at least `min_frames`. |
| `reference(arr, frames)` | Mean at the reference frames. |
| `normalize(arr, frames)` | Subtraction of the reference, for Y. Returns `(delta, ref)`. |
| `channel_stats(refs)` | Typical \|reference\| and usual sign of each channel, from training recordings. |
| `bad_channels(ref, typical, sign, frac)` | Channels with a too small or flipped reference. |
| `normalize_relative(x, frames, typical, sign)` | `(x - ref) / ref`, bad channels set to 0, for X. |
| `pixel_rms(y)` | RMS of all the pixels of a normalised recording (training filter, `TRAINING_RMS_MAX`). |
| `StandardisedRidge(alpha)` | Inside the model: standardisation of X and Y + ridge, `fit`, `predict`. |
| `fit_model(recordings, alpha)` | The whole model from raw `(x, y)` recordings: reference, relative change, altered channels, ridge. |
| `EITReconstructionModel` | `predict(x)` for one whole recording, `save`, `load` (ridge + channel statistics, one `.npz`). |
| `reconstruct(raw, model, lowpass_bpm, like)` | `.eit` to `ReconstructedFrameData`; `like` (the `.bin` of the same acquisition) gives frame times, events and ventilator signals. |
| `lowpass(images, fs, cutoff_bpm)` | Zero-phase Butterworth low-pass along time, 50/min. |

### `metrics`

| Function | Returns | Meaning |
|----------|---------|---------|
| `mse_per_frame(Y_true, Y_pred)` | `(N,)` | Per-frame error |
| `rmse_global(Y_true, Y_pred)` | scalar | Overall error in pixel units |
| `r2_score(Y_true, Y_pred)` | scalar | Fraction of variance explained, pooled over frames x pixels |
| `error_map(Y_true, Y_pred)` | `(32, 32)` | Mean absolute error per pixel |
| `correlation_per_frame(Y_true, Y_pred)` | `(N,)` | Spatial pattern fidelity per frame |
| `global_signal_correlation(Y_true, Y_pred)` | scalar | Waveform fidelity of the global signal |
| `summary_metrics(Y_true, Y_pred)` | dict | All of the above in one call |

`r2_score` is pooled over all elements around the grand mean, not the per-pixel average of
sklearn's `r2_score(multioutput="uniform_average")`.

## Dataset

`python -m fasteit.reconstruction.build_dataset all --root /path/to/eit` scans the patient folders,
checks the frame counts of each pair, writes one `.npz` per patient (`X_vv`, `X_raw`, `Y`, `rec_id`)
and a sample of `--max-frames` frames per patient drawn at random within each recording.

## Results

The results on our recordings (leave-one-patient-out, validation on a new study, agreement of the
clinical indices) are kept private for now, as the trained models.
