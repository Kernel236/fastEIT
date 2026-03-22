"""
Core data models for fasteit.

- BaseData               — base container for raw parsed data
- ReconstructedFrameData — pixel matrices + synchronized signals (.bin, .txt)
- ContinuousSignalData   — signal-only tables, no matrices (.asc)
- RawImpedanceData       — raw transimpedances for pyEIT (.eit, .x)
"""

from .base_data import BaseData
from .continuous_data import ContinuousSignalData
from .raw_impedance_data import RawImpedanceData
from .reconstructed_data import ReconstructedFrameData

__all__ = [
    "BaseData",
    "ReconstructedFrameData",
    "ContinuousSignalData",
    "RawImpedanceData",
]
