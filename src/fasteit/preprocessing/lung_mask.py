"""
Lung mask generation — identify ventilated pixels in the 32x32 EIT image.

A pixel is considered "ventilated" if its tidal impedance variation (TIV)
exceeds a threshold (default: 20% of max TIV, or Otsu automatic).

References:
    Zhao 2010: DOI (see fastEIT_reference_DEFINITIVO.md)
    TREND 2017: DOI 10.1136/thoraxjnl-2016-208357

TODO (Task 4.4.1): lung_mask_threshold(tiv_map, threshold_pct=0.20) -> np.ndarray bool 32x32
TODO (Task 4.4.2): lung_mask_otsu(tiv_map) -> np.ndarray bool
TODO (Task 4.7.1): apply_morphological_closing(mask) -> np.ndarray bool
TODO (Task 4.7.2): remove_small_components(mask, min_pixels) -> np.ndarray bool
TODO (Task 4.5.1): lung_mask_laem (post-v1, Zhao 2016)
TODO (Task 4.6.1): lung_mask_watershed (post-v1, Somhorst 2026)
"""
