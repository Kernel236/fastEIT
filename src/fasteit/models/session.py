"""
RawData dataclass — output of all parsers, input to preprocessing.

TODO (Task 0.5.2): Implement RawData dataclass with:
    frames: np.ndarray          shape (N_frames, 32, 32) — impedance images
    timestamps: np.ndarray      shape (N_frames,) — seconds from start
    metadata: dict              sample_rate, patient_id, recording_date, ...
    sample_rate: float          frames per second (default 20.0 for PulmoVista)
    medibus: dict | None        pressures, volumes, flows, PEEP (if available)
    event_markers: list[tuple]  (frame_idx, event_text)
"""
