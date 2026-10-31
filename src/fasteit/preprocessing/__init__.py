"""Preprocessing layer (layer 2): from parsed frames to a recording ready for analysis.

Entry point: ``preprocess(raw, config, exclusions, context) -> EITRecording``.

One module per step, in pipeline order:
- cardiac_filter   - respiratory rate, cardiac filter (MDN) or moving average
- exclusions       - device and manual exclusions, and the breaths they touch
- breaths          - breath detection on the global signal
- breath_quality   - quality flags of each breath
- lung_mask        - watershed lung mask from the valid breaths
- regions          - regions of interest inside the lung mask
- auxiliary        - pressures, flow, volume and device markers, "no data" values as
                     NaN, the same units for every vendor
- breath_pressures - PEEP and plateau of each breath on the airway pressure
- pipeline         - preprocess(), all the steps in a fixed order

Evidence for every choice: notebooks/05_preprocessing_choices.ipynb (pressures of each
breath: notebooks/06_peep_trial_segmentation.ipynb).
"""

from .pipeline import preprocess

__all__ = ["preprocess"]
