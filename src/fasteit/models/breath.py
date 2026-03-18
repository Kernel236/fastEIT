"""
Breath dataclass — one respiratory cycle with clinical features.

Migrated and refactored from Flask backend (Task 3.1.x).

TODO (Task 0.5.3): Define Breath dataclass with:
    frame_start: int        index in RawData.frames
    frame_peak: int         peak inspiratory frame
    frame_end: int          end-expiratory frame
    quality_flag: str       "ok" | "short_breath" | "artifact" | ...
    peep_step_id: int | None
    # Features populated by Fase 5:
    eeli: float | None
    tiv: float | None
    gi_index: float | None
    cov: float | None
    ti_s: float | None      inspiratory time (seconds)
    te_s: float | None      expiratory time
    ttot_s: float | None    total breath duration
    ti_ttot: float | None   Ti/Ttot ratio
    # Regional (populated later):
    tiv_ventral: float | None
    tiv_dorsal: float | None
    # ...
    def to_dict(self) -> dict: ...
"""
