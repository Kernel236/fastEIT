# fastEIT

Python library for electrical impedance tomography (EIT) recordings of the Dräger
PulmoVista 500 and of Timpel: it reads the files, prepares each breath and computes the
clinical indices of tidal recordings and PEEP trials, as tables ready for statistical analysis.

!!! warning "Research use only"
    fastEIT is for research. It is not a medical device and has not been validated for
    clinical decisions.

## From a file to the clinical indices

Each layer returns one object that holds the object of the layer before it:

```
Dräger .bin, Timpel .csv / .txt / .asc
 └─ load_data() ───────────► ReconstructedFrameData   layer 1: the file as it is (32x32 images)
     └─ preprocess() ──────► EITRecording             layer 2: the measured breaths
          ├─ tidal_features() ───────► TidalFeatures       layer 3: indices
          └─ segment_peep_trial() ───► PeepTrial           layer 2b: PEEP steps
               └─ peep_trial_features() ► PeepTrialFeatures  layer 3: indices
                    └─ recording_tables() ► tables, CSV        layer 4: export
```

The pipeline starts from reconstructed images. A Dräger `.eit` file holds the raw
measurements (`RawImpedanceData`, 208 transimpedances per frame) and needs a
reconstruction first; a Dräger `.asc` export holds signals only (`ContinuousSignalData`)
and has no images ([parsers](parsers.md)).

| Layer | What it does | How it works |
|-------|--------------|-------|
| 1. Parsers | Dräger `.bin`, `.eit`, `.asc` and Timpel files to data containers | [Parsers](parsers.md) |
| 2. Preprocessing | cardiac filter, breaths with quality flags, exclusions, lung mask, regions; auxiliary signals in cmH2O, L/s, mL; PEEP, plateau, driving pressure and tidal volume of each breath | [Preprocessing](preprocessing.md) |
| 2b. Segmentation | PEEP steps from the pressure or placed by hand, breaths analysed, driving pressure and tidal volume of each step | [Segmentation](segmentation.md) |
| 3. Clinical indices | EELI, TIV, share by region, GI, centre of ventilation, compliance, Costa collapse and overdistension, changes between steps, recruitment-to-inflation ratio | [Clinical indices](features.md) |
| 4. Export | one row per recording, breath and PEEP step, keyed by patient, recording and phase; a whole study at once; CSV ready for statistical analysis | [Export](export.md) |

Every method states its source (paper and DOI, or the evidence notebook) and the
reimplemented steps are checked against [eitprocessing](https://github.com/EIT-ALIVE/eitprocessing).

Start with the **user guide**: [install and supported files](getting_started.md), then
[analyse a recording](guide/analyse.md), [cut a recording](guide/segments.md), check it
with the [review figures](review.md) and [export the tables](export.md). The pages under
"How it works" describe each layer and its choices; every function is in the
[API reference](api/index.md).
