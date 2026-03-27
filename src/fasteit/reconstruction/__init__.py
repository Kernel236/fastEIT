"""Data-driven EIT image reconstruction.

Learns the Dräger PulmoVista 500 reconstruction mapping directly from
paired .eit / .bin recordings, bypassing the need for the proprietary
Newton-Raphson FEM algorithm and thorax-shaped mesh.

Device-specific: trained and validated on Dräger PulmoVista 500 data only
(16 electrodes, adjacent drive, 208 measurements per frame).
"""
