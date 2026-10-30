"""Regions of interest inside the lung mask.

Orientation of Draeger images (PulmoVista 500 instructions for use): seen from the
feet, like a CT slice. Row 0 is ventral; column 0 is the patient's RIGHT, column 31
the patient's left. Consistent with the power at the heart rate per pixel
(notebooks/05_preprocessing_choices.ipynb, section 9). Verified on Draeger only.

Regions (section 9):
- ventral/dorsal halves and four layers are cut on the vertical extent of the lung
  (first to last row of the mask), not on the image, so "dorsal" is always the
  dorsal part of that lung wherever the lungs sit in the image. Same approach as
  the ``createSummersROIs.m`` script of the original MATLAB analysis; the published
  source of this approach is still to be found;
- right/left halves are split at the image midline (between columns 15 and 16),
  the sternum-spine axis with the belt centred on the patient;
- quadrants = ventral/dorsal x right/left;
- every region is intersected with the lung mask.

Limit: these regions are relative to the VENTILATED lung. If the dorsal lung
collapses, its pixels leave the mask and "dorsal" becomes the dorsal half of what is
still ventilated, so the ventral/dorsal split hides part of a ventral shift of
ventilation.
Indices meant to show a shift of ventilation (centre of ventilation, dorsal fraction,
silent spaces) must be computed in image coordinates, not on these regions.

Not the same as ``ReconstructedFrameData.roi_signals`` (layer 1), which are four
fixed strips of 8 image rows each.
"""

from __future__ import annotations

import numpy as np


def lung_regions(mask: np.ndarray) -> dict[str, np.ndarray]:
    """All the regions of interest of a lung mask.

    Args:
        mask: Boolean lung mask (rows, cols), row 0 ventral, column 0 patient right.

    Returns:
        Dict of boolean masks, each inside ``mask``: ``ventral``, ``dorsal``,
        ``right``, ``left``, ``layer_1`` (most ventral) to ``layer_4`` (most
        dorsal), ``ventral_right``, ``ventral_left``, ``dorsal_right``,
        ``dorsal_left``.

    Raises:
        ValueError: if the mask is empty.
    """
    mask = np.asarray(mask, dtype=bool)
    rows_with_lung = np.flatnonzero(mask.any(axis=1))
    if len(rows_with_lung) == 0:
        raise ValueError("Empty lung mask: no regions can be defined.")
    first_row = rows_with_lung[0]
    lung_height = rows_with_lung[-1] - first_row + 1

    # Position of every pixel: rows counted from the top of the lung.
    n_rows, n_cols = mask.shape
    row, col = np.indices((n_rows, n_cols))
    row_in_lung = row - first_row

    ventral = row_in_lung * 2 < lung_height
    layer = np.clip(row_in_lung * 4 // lung_height, 0, 3)
    right = col < n_cols // 2  # image left = patient right

    regions = {
        "ventral": ventral,
        "dorsal": ~ventral,
        "right": right,
        "left": ~right,
        "layer_1": layer == 0,
        "layer_2": layer == 1,
        "layer_3": layer == 2,
        "layer_4": layer == 3,
        "ventral_right": ventral & right,
        "ventral_left": ventral & ~right,
        "dorsal_right": ~ventral & right,
        "dorsal_left": ~ventral & ~right,
    }
    return {name: region & mask for name, region in regions.items()}
