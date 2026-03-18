"""
Configuration dataclasses for fasteit pipeline parameters.

All default values must be evidence-based with literature citations.

TODO (Task 0.6.1): Implement Config dataclass with sections:
    - parsing: frame_size, endianness
    - filtering: cutoff_hz, order, filter_type
    - lung_mask: threshold_method, threshold_pct
    - breath_detection: min_duration_s, max_duration_s, min_prominence
    - peep: smoothing_window, step_threshold, min_duration
    - features: roi_method
    - export: format
TODO (Task 0.6.2): Add validated default values with citations
TODO (Task 0.6.3): Add preset_sensitive and preset_permissive
"""
