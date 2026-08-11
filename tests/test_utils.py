import logging
import random

import numpy as np

from src.utils import get_logger, set_seed


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
