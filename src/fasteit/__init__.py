"""
fasteit - EIT data parsing, preprocessing, and clinical feature extraction.

Python library for parsing Dräger PulmoVista files (.bin, .eit, .txt),
preprocessing EIT signals, and extracting breath-level clinical features
into analysis-ready datasets.

Disclaimer: This software is for research purposes only. Not a medical device.
"""

import logging

# A library only emits log messages; the script or application decides where they go
# (e.g. logging.basicConfig(level=logging.INFO)). The warnings are also kept in
# EITRecording.log.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0.dev0"
__author__ = "Riccardo Del Signore"
__license__ = "Apache-2.0"

__all__ = ["__version__"]
