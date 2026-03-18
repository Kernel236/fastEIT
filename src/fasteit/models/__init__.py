"""
Core data models for fasteit.

- BaseData      — base container for raw parsed data
- BinData       — output of BinParser (.bin, reconstructed 32×32 images)
"""

from .base_data import BaseData
from .bin_data import BinData

__all__ = ["BaseData", "BinData"]
