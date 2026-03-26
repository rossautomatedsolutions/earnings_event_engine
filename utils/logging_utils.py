"""Logging helpers for project scripts and modules."""

import logging


def get_logger(name: str = "earnings_event_engine") -> logging.Logger:
    """Create or return a standard console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
