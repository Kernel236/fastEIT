"""Tests for RunLog: the messages of one run, each with its level."""

import logging

import pytest

from fasteit.run_log import RunLog, split_level


def test_messages_keep_their_order_with_a_prefix_for_checks_and_warnings():
    log = RunLog(logging.getLogger("test_run_log"))
    log.ok("PEEP read for 131 of 131 breaths.")
    log.info("131 breaths.")
    log.warning("No tidal volume.")
    assert log.messages == [
        "OK: PEEP read for 131 of 131 breaths.",
        "131 breaths.",
        "WARNING: No tidal volume.",
    ]


@pytest.mark.parametrize(
    ("message", "level", "text"),
    [
        (
            "OK: PEEP read for 131 of 131 breaths.",
            "ok",
            "PEEP read for 131 of 131 breaths.",
        ),
        ("131 breaths.", "info", "131 breaths."),
        ("WARNING: No tidal volume.", "warning", "No tidal volume."),
    ],
)
def test_split_level(message, level, text):
    assert split_level(message) == (level, text)


def test_a_check_goes_to_the_python_logger_as_info(caplog):
    log = RunLog(logging.getLogger("test_run_log"))
    with caplog.at_level(logging.INFO, logger="test_run_log"):
        log.ok("a check passed")
    assert caplog.records[-1].levelno == logging.INFO
    assert caplog.records[-1].getMessage() == "a check passed"
