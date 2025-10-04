"""프로젝트 전역에서 사용 가능한 로거 설정 모듈."""

from __future__ import annotations

import logging
import os
from typing import Optional


LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
DEBUG_ENABLED = LOG_LEVEL == "DEBUG"


_LOGGER: Optional[logging.Logger] = None


def get_app_logger() -> logging.Logger:
    """기본 애플리케이션 로거를 반환한다."""

    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("momentum_etf")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(
            LOG_LEVEL if LOG_LEVEL in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"
        )
        logger.propagate = False

    _LOGGER = logger
    return logger


def is_debug_enabled() -> bool:
    """DEBUG 로그 출력 여부를 반환한다."""

    return DEBUG_ENABLED


__all__ = ["get_app_logger", "is_debug_enabled"]
