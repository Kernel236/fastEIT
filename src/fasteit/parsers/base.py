"""Abstract base class for all fasteit parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fasteit.models import BaseData


class BaseParser(ABC):
    """Abstract base class for file parsers.

    Concrete subclasses must implement:
        - parse(path)    — read the file and return a data container
        - validate(path) — check that the file is in the expected format

    The recommended entry point is parse_safe(), which validates before parsing.
    """

    @abstractmethod
    def parse(self, path: Path) -> BaseData:  # noqa: F821
        """Parse the file and return raw data.

        Args:
            path: Path to the file to parse.

        Returns:
            A BaseData subclass instance with the file contents.
        """
        ...

    @abstractmethod
    def validate(self, path: Path) -> bool:
        """Return True if the file is in the expected format.

        Args:
            path: Path to the file to validate.
        """
        ...

    def parse_safe(self, path: Path) -> BaseData:  # noqa: F821
        """Validate then parse. Recommended entry point.

        Args:
            path: Path to the file.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the file fails format validation.

        Returns:
            A BaseData subclass instance with the file contents.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not self.validate(path):
            raise ValueError(f"File invalid or unsupported format: {path}")
        return self.parse(path)
