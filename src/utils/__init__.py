"""
Centralized logging configuration for the 3D reconstruction pipeline.

Usage:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import os
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
):
    """
    Configure the root logger for the entire pipeline.

    Call once at program startup (e.g., in train.py / evaluate.py).

    Args:
        level:    Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to write logs to.
        console:  Whether to log to stdout.
    """
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Args:
        name: Logger name (typically __name__).
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
