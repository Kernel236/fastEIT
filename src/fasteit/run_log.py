"""RunLog: the messages of one run of a pipeline (preprocess, segment_peep_trial).

The messages are kept with the result (e.g. ``EITRecording.log``) and also sent to the
Python logger of the module that runs, which the application configures.
"""

from __future__ import annotations

import logging


class RunLog:
    """Collects the messages of one run and sends them to ``logger``."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.messages.append(f"WARNING: {message}")
        self.logger.warning(message)
