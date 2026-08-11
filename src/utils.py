"""Small helpers shared by the training and evaluation stages."""

from __future__ import annotations

import logging
import random

import numpy as np

from src.config import CONFIG

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a console logger, attaching its handler only once."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT))

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Without this the root logger prints every record a second time.
        logger.propagate = False

    return logger


def set_seed(seed: int = CONFIG.random_seed) -> int:
    """Seed every source of randomness used in this project.

    Torch is seeded only when it is installed, so the tree-based models keep
    working in an environment without the deep learning dependencies.

    PYTHONHASHSEED is deliberately not set here: the interpreter reads it at
    start-up, so assigning it from inside a running process does nothing.
    """

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed
