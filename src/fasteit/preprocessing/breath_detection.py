"""
Breath detection on global impedance signal.

Identifies inspiratory peaks and expiratory valleys using
scipy.signal.find_peaks with adaptive prominence and distance
parameters based on expected respiratory rate range.

TODO (Task 4.10.1): prefilter_for_detection(signal, fs) -> np.ndarray
    Band-pass 0.1-1.5 Hz before peak detection
TODO (Task 4.10.2): find_inspiratory_peaks(signal, config) -> np.ndarray indices
TODO (Task 4.10.3): find_expiratory_valleys(signal, config) -> np.ndarray indices
TODO (Task 4.10.4): assemble_breaths(peaks, valleys) -> list[tuple[int,int,int]]
    Each tuple: (idx_start, idx_peak, idx_end)
TODO (Task 4.11.1): compute_quality_flags(breath, signal) -> str
    Flags: ok | short_breath | long_breath | artifact | low_signal |
           suspect_ti_ratio | negative_delta_z
"""
