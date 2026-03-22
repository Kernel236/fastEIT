"""Timpel parser placeholder.

Scaffold only: parser implementation will be added in a dedicated task.
"""

from __future__ import annotations

from pathlib import Path

from fasteit.models.reconstructed_data import ReconstructedFrameData
from fasteit.parsers.base import BaseParser


class TimpelTabularParser(BaseParser):
    """Placeholder parser for Timpel tabular recordings (.csv/.txt/.asc)."""

    def validate(self, path: Path) -> bool:
        """Template validation hook used by `parse_safe`."""
        path = Path(path)
        return path.exists() and path.stat().st_size > 0

    def parse(
        self,
        path: Path,
        first_frame: int = 0,
        max_frames: int | None = None,
    ) -> ReconstructedFrameData:
        """Parse Timpel tabular file into ReconstructedFrameData.

        Timpel tabular format: one row per frame, 1030 columns total.
        - cols 0-1023: reconstructed 32x32 pixel matrix (row-major)
        - cols 1024-1028: 5 continuous signals (device-specific, see timpel_dtypes)

        TODO(Task 3.x):
        - load tabular matrix (CSV/TXT-like, tab or comma separated)
        - enforce 1030-column schema, raise on mismatch
        - reshape cols 0-1023 to (N, 32, 32)
        - map cols 1024-1028 via timpel_dtypes field names
        - build and return ReconstructedFrameData
        """
        _ = (Path(path), first_frame, max_frames)
        raise NotImplementedError(
            "TimpelTabularParser.parse() scaffold ready. Implement Task 3.x."
        )
