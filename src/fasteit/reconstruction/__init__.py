"""Data-driven EIT image reconstruction.

Learns the Dräger PulmoVista 500 reconstruction mapping directly from
paired .eit / .bin recordings, bypassing the need for the proprietary
Newton-Raphson FEM algorithm and thorax-shaped mesh.

Device-specific: trained and validated on Dräger PulmoVista 500 data only
(16 electrodes, adjacent drive, 208 measurements per frame).

Requires the ``[reconstruction]`` extra::

    pip install fasteit[reconstruction]
"""

from fasteit.reconstruction.data_prep import load_paired, normalize, prepare_dataset
from fasteit.reconstruction.metrics import summary_metrics
from fasteit.reconstruction.ridge_model import RidgeReconstructor

__all__ = [
    "RidgeReconstructor",
    "load_paired",
    "normalize",
    "prepare_dataset",
    "summary_metrics",
]
