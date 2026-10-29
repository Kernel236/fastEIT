"""Parts of a recording excluded from the analysis.

Exclusions come from the user (``source="manual"``) or from the device
(``source="device"``: frames with a timing error or with missing pixel data). They
are widened by a margin, because the forward-backward filters spread an artifact
over the neighbouring samples, and a breath that touches an exclusion is marked
``excluded``: it stays in the breath table (fastEIT design decision D5).

Frames with missing pixel data are also filled with the nearest valid frame, so that
the filters can run on a continuous signal; the filled frames are excluded.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from fasteit.models.eit_recording import Segment
from fasteit.preprocessing.breaths import fill_with_nearest


def frames_to_segments(
    time: np.ndarray, bad_frames: np.ndarray, fs: float, label: str
) -> list[Segment]:
    """One device exclusion per run of consecutive bad frames.

    Args:
        time:       Time of each frame in seconds from the start of the recording.
        bad_frames: Boolean array, True for the frames to exclude.
        fs:         Sampling frequency in Hz; a run ends one sample after its last
                    frame, so that a single bad frame gives a segment of 1/fs.
        label:      Why the frames are excluded, e.g. ``"timing_error"``.

    Returns:
        List of segments with ``source="device"``.
    """
    bad = np.flatnonzero(bad_frames)
    if len(bad) == 0:
        return []
    runs = np.split(bad, np.flatnonzero(np.diff(bad) != 1) + 1)
    return [
        Segment(
            float(time[run[0]]),
            float(time[run[-1]] + 1 / fs),
            label=label,
            source="device",
        )
        for run in runs
    ]


def missing_frames(pixels: np.ndarray) -> np.ndarray:
    """Frames where some pixel has no value (NaN or inf).

    A pixel that has no value in the whole recording (e.g. outside the belt) does not
    make a frame missing: it is simply not part of the image.

    Args:
        pixels: Array (N, rows, cols).

    Returns:
        Boolean array (N,).
    """
    not_finite = ~np.isfinite(pixels)
    always_missing = not_finite.all(axis=0)
    return (not_finite & ~always_missing).any(axis=(1, 2))


def fill_missing_frames(pixels: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Replace the missing frames with the nearest valid frame.

    Same rule as ``breaths.fill_with_nearest``, applied to whole frames.

    Args:
        pixels:  Array (N, rows, cols).
        missing: Boolean array (N,) from ``missing_frames``.

    Returns:
        A copy of ``pixels``.

    Raises:
        ValueError: if every frame is missing.
    """
    frame_numbers = np.arange(len(pixels), dtype=float)
    source = fill_with_nearest(frame_numbers, np.flatnonzero(missing)).astype(int)
    return pixels[source]


def _label_list(label: str) -> list[str]:
    """``"artifact, movement"`` -> ``["artifact", "movement"]``; empty labels dropped."""
    return [name for name in label.split(", ") if name]


def widen(segments: list[Segment], margin: float, duration: float) -> list[Segment]:
    """Each segment made longer by ``margin`` seconds on both sides.

    The result is clipped to the recording, from 0 to ``duration`` seconds.

    Raises:
        ValueError: if a segment starts after the end of the recording.
    """
    for segment in segments:
        if segment.start >= duration:
            raise ValueError(
                f"Exclusion {segment.start:g}-{segment.end:g} s starts after the end "
                f"of the recording ({duration:.1f} s)."
            )
    return [
        dataclasses.replace(
            segment,
            start=max(0.0, segment.start - margin),
            end=min(duration, segment.end + margin),
        )
        for segment in segments
    ]


def merge_overlapping(segments: list[Segment]) -> list[Segment]:
    """Join the segments of the same source that overlap or touch.

    Manual and device segments are never joined, so a manual exclusion stays apart
    from the device ones (both are widened by the margin before). The labels of
    joined segments are listed together, e.g. ``"artifact, movement"``; the
    parameters are those of the first segment.

    Returns:
        Segments sorted by start time.
    """
    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda s: (s.source, s.start)):
        last = merged[-1] if merged else None
        if last and last.source == segment.source and segment.start <= last.end:
            labels = _label_list(last.label)
            labels += [
                name for name in _label_list(segment.label) if name not in labels
            ]
            merged[-1] = dataclasses.replace(
                last, end=max(last.end, segment.end), label=", ".join(labels)
            )
        else:
            merged.append(segment)
    return sorted(merged, key=lambda s: s.start)


def breaths_in_segments(
    breath_starts: np.ndarray, breath_ends: np.ndarray, segments: list[Segment]
) -> tuple[np.ndarray, list[str | None]]:
    """Which breaths touch an exclusion, and why.

    Args:
        breath_starts: Start time of each breath, in seconds.
        breath_ends:   End time of each breath, in seconds.
        segments:      Exclusions.

    Returns:
        ``excluded``: boolean array (n_breaths,); ``labels``: for each breath the
        labels of the exclusions it touches, e.g. ``"artifact"`` (``"unlabelled"``
        if none has a label), or None if the breath is not excluded.
    """
    excluded = np.zeros(len(breath_starts), dtype=bool)
    labels: list[str | None] = []
    for i, (start, end) in enumerate(zip(breath_starts, breath_ends, strict=True)):
        touched = [s for s in segments if s.overlaps(start, end)]
        excluded[i] = bool(touched)
        names = [name for s in touched for name in _label_list(s.label)]
        if touched:
            labels.append(", ".join(dict.fromkeys(names)) or "unlabelled")
        else:
            labels.append(None)
    return excluded, labels
