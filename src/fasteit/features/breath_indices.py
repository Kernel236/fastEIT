"""Indices of the distribution of ventilation in one breath, from its tidal image.

The tidal image of a breath is each pixel at the end-inspiratory peak minus the same
pixel at the start of the breath (end-expiratory valley), inside the lung mask: the
"inspiratory" TIV of TREND (Frerichs et al., Thorax 2017, DOI:
10.1136/thoraxjnl-2016-208357), one sample at the frames of the layer 2 breaths
(notebooks/07_clinical_indices.ipynb, sections 1-2).
"""

from __future__ import annotations

import numpy as np


def tidal_images(
    pixels: np.ndarray, starts: np.ndarray, peaks: np.ndarray, lung_mask: np.ndarray
) -> np.ndarray:
    """Tidal image of each breath: pixel at the peak minus pixel at the start.

    Args:
        pixels:    (N, rows, cols) images after the cardiac step.
        starts:    Frame of the start of each breath.
        peaks:     Frame of the peak of each breath.
        lung_mask: (rows, cols) bool, the ventilated lung.

    Returns:
        (n_breaths, rows, cols) float, NaN outside the lung mask.
    """
    images = (pixels[peaks] - pixels[starts]).astype(float)
    images[:, ~lung_mask] = np.nan
    return images


def global_inhomogeneity(image: np.ndarray) -> float:
    """Global inhomogeneity index of one tidal image (lung pixels, NaN elsewhere).

    GI = sum(|DI_xy - median(DI_lung)|) / sum(DI_lung), over the lung pixels (Zhao et
    al., Intensive Care Med 2009;35:1900-1906, DOI: 10.1007/s00134-009-1589-y). 0 for a
    homogeneous ventilation; higher when the ventilation is uneven.

    Returns:
        GI; NaN if the lung has no tidal variation.
    """
    lung = image[~np.isnan(image)]
    total = lung.sum()
    if len(lung) == 0 or total <= 0:
        return np.nan
    return float(np.abs(lung - np.median(lung)).sum() / total)


def centre_of_ventilation(image: np.ndarray) -> tuple[float, float]:
    """Centre of ventilation of one tidal image, ventro-dorsal and right-left.

    TIV-weighted mean row (column), divided by the last row (column) of the image
    (Frerichs et al., Am J Respir Crit Care Med 2006;174:772-779, DOI:
    10.1164/rccm.200512-1942OC), on the **image grid**, not on the regions of the lung,
    so that a shift of ventilation is not hidden (notebooks 05 and 07). The image is
    seen from the feet: ventro-dorsal 0 = ventral, 1 = dorsal; right-left 0 = patient
    right, 1 = patient left. 0.5 is the middle of the image.

    Returns:
        (ventro-dorsal, right-left); NaN if the lung has no tidal variation.
    """
    tiv = np.nan_to_num(image)
    total = tiv.sum()
    if total <= 0:
        return np.nan, np.nan
    rows, cols = np.indices(tiv.shape)
    return (
        float((tiv * rows).sum() / total / (tiv.shape[0] - 1)),
        float((tiv * cols).sum() / total / (tiv.shape[1] - 1)),
    )
