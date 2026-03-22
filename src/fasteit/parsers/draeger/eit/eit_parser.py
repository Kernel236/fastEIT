"""Drager `.eit` parser scaffold (Task 2.x)."""

from __future__ import annotations

from pathlib import Path

from fasteit.parsers.base import BaseParser


class DragerEitParser(BaseParser):
    """Parser scaffold for PulmoVista `.eit` files."""

    def validate(self, path: Path) -> bool:
        """Quick format validation.

        TODO(Task 2.3.2):
        - detect Draeger magic string in header ASCII region
        - reject known non-Draeger formats with actionable message
        """
        _ = path
        raise NotImplementedError("Implement Task 2.3.2 format validation")

    def parse(self, path: Path):
        """Parse `.eit` file.

        TODO(Task 2.3.1): parse header ASCII metadata
        TODO(Task 2.4.1): parse binary frame sections
        TODO(Task 2.4.2): compute transimpedance series (208 per frame)
        """
        _ = path
        raise NotImplementedError("Implement Task 2.x DragerEitParser.parse")
