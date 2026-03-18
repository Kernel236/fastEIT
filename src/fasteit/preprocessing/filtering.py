"""
EIT signal filtering — cardiac artifact removal.

The cardiac signal (~1-1.5 Hz) contaminates impedance measurements
in pixels near the heart. A zero-phase low-pass Butterworth filter
(cutoff ~0.5 Hz) attenuates cardiac frequencies while preserving
the respiratory signal (0.1-0.5 Hz).

References:
    Wisse 2024: DOI 10.1186/s40635-024-00686-9

TODO (Task 4.1.1): butterworth_lowpass(signal, cutoff=0.5, order=4, fs) -> np.ndarray
    Uses scipy.signal.butter + sosfiltfilt (zero-phase, stable for high orders)
TODO (Task 4.1.2): butterworth_bandpass, butterworth_highpass (optional)
TODO (Task 4.2.1): mdn_filter (Multiple Digital Notch, post-v1)
TODO (Task 4.3.1): perfusion_bandpass 1-3 Hz (post-v1)
"""
