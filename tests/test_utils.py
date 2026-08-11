import logging
import random

import numpy as np
import pytest

from src.utils import get_logger, set_seed, timed


@pytest.fixture
def captured_records(caplog):
    """Capture from a named logger that deliberately does not propagate.

    get_logger() sets propagate = False so records are not printed twice, which
    also keeps them away from the root handler caplog normally listens on.
    """

    def capture(logger: logging.Logger):
        logger.addHandler(caplog.handler)
        caplog.set_level(logging.INFO, logger=logger.name)

        return caplog

    yield capture


def test_set_seed_makes_random_draws_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand(3).tolist())

    set_seed(123)
    second = (random.random(), np.random.rand(3).tolist())

    assert first == second


def test_set_seed_returns_the_seed_it_applied():
    assert set_seed(7) == 7


def test_get_logger_does_not_stack_handlers():
    logger = get_logger("test.duplicate")
    handler_count = len(logger.handlers)

    get_logger("test.duplicate")

    assert len(logger.handlers) == handler_count == 1
    assert logger.level == logging.INFO


def test_timed_logs_the_elapsed_duration(captured_records):
    logger = get_logger("test.timed")
    caplog = captured_records(logger)

    with timed("training", logger):
        pass

    assert "training took" in caplog.text


def test_timed_reraises_and_still_reports(captured_records):
    logger = get_logger("test.timed.failure")
    caplog = captured_records(logger)

    with pytest.raises(ValueError):
        with timed("broken step", logger):
            raise ValueError("boom")

    assert "broken step took" in caplog.text
