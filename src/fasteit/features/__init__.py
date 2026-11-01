"""Clinical indices layer (layer 3): from preprocessed recordings to feature tables.

Entry points:
- ``tidal_features(recording, phase) -> TidalFeatures``
- ``peep_trial_features(trial) -> PeepTrialFeatures``

- breath_indices - tidal image, global inhomogeneity, centre of ventilation
- costa          - collapse and overdistension, crossing PEEP
- pipeline       - the tables of each breath, region and step

Evidence for every choice: notebooks/07_clinical_indices.ipynb.
"""

from .pipeline import peep_trial_features, tidal_features

__all__ = ["peep_trial_features", "tidal_features"]
