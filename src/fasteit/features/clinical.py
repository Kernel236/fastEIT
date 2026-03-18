"""
Clinical EIT features — breath-level and PEEP-step-level.

All formulas must match definitions in fastEIT_reference_DEFINITIVO.md.
Every function must include a docstring with paper citation and DOI.

TODO (Task 5.1.1): compute_eeli(frames, breath, lung_mask) -> float
    Reference: Frerichs 2017, DOI in reference doc
TODO (Task 5.1.2): compute_tiv(frames, breath, lung_mask) -> float
    Reference: Frerichs 2017
TODO (Task 5.1.3): compute_gi_index(tiv_map) -> float
    GI = sum|DI - median(DI)| / sum|DI|
    Reference: Zhao 2009
TODO (Task 5.1.4): compute_cov(tiv_map) -> float
    Center of ventilation (vertical)
    Reference: Frerichs 2006
TODO (Task 5.1.6): compute_breath_timing(breath, fs) -> dict
    Ti, Te, Ttot, Ti/Ttot, f_resp
TODO (Task 5.2.x): Regional features per ROI (halves, quadrants)
TODO (Task 5.3.x): Pixel-level features (delta-Z, RVD, fEIT)
TODO (Task 5.7.x): PEEP step detection (L1 Medibus, L2 EELI experimental)
"""
