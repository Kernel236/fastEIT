"""
PeepStep dataclass — aggregated features over a stable PEEP level.

TODO (Task 0.5.4): Implement PeepStep dataclass with:
    peep_level: float           cmH2O
    frame_start: int
    frame_end: int
    breath_indices: list[int]   indices into Session.breaths
    stable: bool                False if high variability during step
    # Aggregated features (median, IQR, n):
    tiv_median: float | None
    tiv_iqr: float | None
    eeli_median: float | None
    gi_median: float | None
    # ...
"""
