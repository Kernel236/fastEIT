"""
Core data models for fasteit.

- BaseData               - base container for raw parsed data
- ReconstructedFrameData - pixel matrices + synchronized signals (Dräger .bin, Timpel .csv/.txt/.asc)
- ContinuousSignalData   - signal-only tables, no matrices (.asc)
- RawImpedanceData       - raw transimpedances for pyEIT (.eit)
- EITRecording           - preprocessed recording (layer 2), with Segment and
                           RecordingContext
- PeepTrial              - PEEP trial split into PEEP steps (layer 2b)
- TidalFeatures, PeepTrialFeatures - clinical indices as tables (layer 3)
"""

from .base_data import BaseData
from .continuous_data import ContinuousSignalData
from .eit_recording import EITRecording, RecordingContext, Segment
from .features import PeepTrialFeatures, TidalFeatures
from .peep_trial import PeepTrial
from .raw_impedance_data import RawImpedanceData
from .reconstructed_data import ReconstructedFrameData

__all__ = [
    "BaseData",
    "ReconstructedFrameData",
    "ContinuousSignalData",
    "RawImpedanceData",
    "EITRecording",
    "RecordingContext",
    "Segment",
    "PeepTrial",
    "TidalFeatures",
    "PeepTrialFeatures",
]
