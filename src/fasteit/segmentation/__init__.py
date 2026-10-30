"""Segmentation layer (layer 2b): splits a preprocessed recording by protocol.

PEEP trial: PEEP steps from the airway pressure (or placed by hand), the pressures of
each breath and the breaths analysed in each step.

Entry point: ``segment_peep_trial(recording, config, peep_steps) -> PeepTrial``.

- breath_pressures - start of inspiration, PEEP and plateau of each breath
- peep_steps       - steps from the pressure or placed by hand, breaths analysed
- pipeline         - segment_peep_trial(), all the steps in a fixed order

Evidence for every choice: notebooks/06_peep_trial_segmentation.ipynb.
"""

from .pipeline import segment_peep_trial

__all__ = ["segment_peep_trial"]
