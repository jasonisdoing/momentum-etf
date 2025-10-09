"""신호 관련 로그를 담당하는 독립형 로거."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

_SIGNAL_LOGGER = None


def get_signal_logger() -> logging.Logger:
    global _SIGNAL_LOGGER
    if _SIGNAL_LOGGER:
        return _SIGNAL_LOGGER

    logger = logging.getLogger("signal.detail")
    if not logger.handlers:
        # 프로젝트 루트 경로(momentum-etf/)를 기준으로 로그 디렉터리를 찾는다.
        project_root = Path(__file__).resolve().parents[2]
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        if ZoneInfo is not None:
            try:
                now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
            except Exception:
                now_kst = datetime.now()
        else:
            now_kst = datetime.now()

        log_path = log_dir / f"{now_kst.strftime('%Y-%m-%d')}.log"

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    _SIGNAL_LOGGER = logger
    return logger


__all__ = ["get_signal_logger"]
