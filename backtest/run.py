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

from config import BACKTEST_CONFIG
from backtest.engine import run_backtest
from utils.env import load_env_if_present

load_env_if_present()

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> int:
    # 인자가 없으면 설정된 모든 종목풀을 순차적으로 실행
    if len(argv) < 2:
        from utils.settings_loader import list_available_ticker_types

        ordered_pools = list_available_ticker_types()
        # BACKTEST_CONFIG에 설정된 것만 필터링하여 실행 (위험 방지)
        pools = [p for p in ordered_pools if p in BACKTEST_CONFIG]
        
        logger.info("모든 종목풀에 대해 지정된 순서대로 백테스트를 실행합니다: %s", pools)
        for pool_id in pools:
            run_backtest(pool_id, BACKTEST_CONFIG)
        return 0

    pool_id = argv[1].strip().lower()
    if pool_id not in BACKTEST_CONFIG:
        logger.error("알 수 없는 pool_id: %s", pool_id)
        logger.error("사용 가능한 pool_id: %s", sorted(BACKTEST_CONFIG.keys()))
        return 2

    run_backtest(pool_id, BACKTEST_CONFIG)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
