"""백테스트 CLI 엔트리 포인트."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from backtest.config import BACKTEST_CONFIG
from backtest.engine import run_backtest
from utils.env import load_env_if_present

load_env_if_present()

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        logger.error("Usage: python backtest/run.py <pool_id>")
        logger.error("사용 가능한 pool_id: %s", sorted(BACKTEST_CONFIG.keys()))
        return 2

    pool_id = argv[1].strip().lower()
    run_backtest(pool_id, BACKTEST_CONFIG)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
