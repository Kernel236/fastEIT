"""Preprocessing layer (layer 2): from parsed frames to a recording ready for analysis.

One module per step, in pipeline order:
- cardiac_filter - remove (or not) the cardiac component from the pixel signals
"""
