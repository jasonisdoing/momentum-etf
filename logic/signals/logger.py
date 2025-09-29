"""Logger wrapper for signals logic."""
from __future__ import annotations
import logging
from signals import get_signal_logger as _root_get_signal_logger


def get_signal_logger() -> logging.Logger:
    return _root_get_signal_logger()


__all__ = ["get_signal_logger"]
