"""Drager-specific parsers namespace.

Future vendor parsers (e.g., timpel) can be added alongside this package.
"""

__all__ = ["DragerAscParser", "DragerBinParser", "DragerEitParser"]


def __getattr__(name: str):
    """Lazy-import parser classes to avoid package import cycles."""
    if name == "DragerAscParser":
        from .asc.asc_parser import DragerAscParser

        return DragerAscParser
    if name == "DragerBinParser":
        from .bin.bin_parser import DragerBinParser

        return DragerBinParser
    if name == "DragerEitParser":
        from .eit.eit_parser import DragerEitParser

        return DragerEitParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
