"""
Parsers for Dräger PulmoVista file formats.

- BaseParser — abstract base class (ABC) for all parsers
- BinParser  (.bin) — reconstructed 32×32 images, Task 1.x
- EitParser  (.eit) — raw electrode voltages, Task 2.x
- CsvParser  (.txt/.csv) — breath-level summary, Task 3.x
"""

from .base import BaseParser

__all__ = ["BaseParser"]
