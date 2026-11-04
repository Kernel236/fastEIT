"""Data-driven EIT image reconstruction.

Learns the Dräger PulmoVista 500 reconstruction mapping directly from
paired .eit / .bin recordings, bypassing the need for the proprietary
Newton-Raphson FEM algorithm and thorax-shaped mesh.

Device-specific: trained and validated on Dräger PulmoVista 500 data only
(16 electrodes, adjacent drive, 208 measurements per frame).

Requires the ``[reconstruction]`` extra::

    pip install fasteit[reconstruction]
"""

from fasteit.reconstruction.data_prep import (
    channel_stats,
    load_paired,
    normalize,
    normalize_relative,
    reference,
    reference_frames,
)
from fasteit.reconstruction.metrics import summary_metrics
from fasteit.reconstruction.model import EITReconstructionModel, fit_model
from fasteit.reconstruction.postprocess import lowpass
from fasteit.reconstruction.reconstruct import reconstruct

__all__ = [
    "EITReconstructionModel",
    "channel_stats",
    "fit_model",
    "load_paired",
    "lowpass",
    "normalize",
    "normalize_relative",
    "reconstruct",
    "reference",
    "reference_frames",
    "summary_metrics",
]
