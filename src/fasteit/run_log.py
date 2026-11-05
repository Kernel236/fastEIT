"""RunLog: the messages of one run of a pipeline (preprocess, segment_peep_trial).

The messages are kept with the result (e.g. ``EITRecording.log``) and also sent to the
Python logger of the module that runs, which the application configures. Each message
has a level, written as a prefix:

- ``"OK: ..."``: a check that passed (no missing frames, every step on a level of the
  protocol, ...);
- no prefix: what was done and with which data (counts, sources, settings);
- ``"WARNING: ..."``: a check that failed, or a missing input with its consequence.

A recording with everything it needs and no problem has no warning.
"""

from __future__ import annotations

import logging

PREFIXES = {"ok": "OK: ", "warning": "WARNING: "}


class RunLog:
    """Collects the messages of one run and sends them to ``logger``."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.messages: list[str] = []

    def ok(self, message: str) -> None:
        """A check that passed."""
        self.messages.append(PREFIXES["ok"] + message)
        self.logger.info(message)

    def info(self, message: str) -> None:
        """What was done, and with which data."""
        self.messages.append(message)
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """A check that failed, or a missing input with its consequence."""
        self.messages.append(PREFIXES["warning"] + message)
        self.logger.warning(message)


def split_level(message: str) -> tuple[str, str]:
    """The level of a logged message (``"ok"``, ``"info"`` or ``"warning"``) and its text.

    Args:
        message: A message of a ``log`` list, e.g. ``"WARNING: No tidal volume."``.

    Returns:
        The level and the message without its prefix.
    """
    for level, prefix in PREFIXES.items():
        if message.startswith(prefix):
            return level, message[len(prefix) :]
    return "info", message
