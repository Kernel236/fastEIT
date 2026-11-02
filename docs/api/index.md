# API reference

Generated from the docstrings of the code, layer by layer. The entry points:

| Layer | Function | Returns |
|-------|----------|---------|
| 1 | [`load_data`](parsers.md#fasteit.parsers.loader.load_data) | `ReconstructedFrameData`, `ContinuousSignalData` or `RawImpedanceData` |
| 2 | [`preprocess`](preprocessing.md#fasteit.preprocessing.pipeline.preprocess) | `EITRecording` |
| 2b | [`segment_peep_trial`](segmentation.md#fasteit.segmentation.pipeline.segment_peep_trial) | `PeepTrial` |
| 3 | [`tidal_features`](features.md#fasteit.features.pipeline.tidal_features), [`peep_trial_features`](features.md#fasteit.features.pipeline.peep_trial_features) | `TidalFeatures`, `PeepTrialFeatures` |
| 4 | [`recording_tables`](export.md#fasteit.export.tables.recording_tables), [`combine_tables`](export.md#fasteit.export.tables.combine_tables), [`write_csv`](export.md#fasteit.export.tables.write_csv) | tables (`dict` of DataFrames), CSV files |

The parameters a study may change are in the [configuration](config.md); the fixed details
of each method are constants of its module, with their source.
