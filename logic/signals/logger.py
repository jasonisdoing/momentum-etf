"""Signal logger (self-contained).

Writes detailed signal logs to project-root `logs/YYYY-MM-DD.log` without
depending on the root `signals.py`.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_SIGNAL_LOGGER = None


def get_signal_logger() -> logging.Logger:
    global _SIGNAL_LOGGER
    if _SIGNAL_LOGGER:
        return _SIGNAL_LOGGER

    logger = logging.getLogger("signal.detail")
    if not logger.handlers:
        # project root: momentum-etf/
        project_root = Path(__file__).resolve().parents[2]
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    _SIGNAL_LOGGER = logger
    return logger


__all__ = ["get_signal_logger"]
