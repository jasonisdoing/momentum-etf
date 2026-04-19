"""백테스트 CLI 엔트리 포인트."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtest.config import BACKTEST_CONFIG
from backtest.engine import run_backtest
from utils.env import load_env_if_present

load_env_if_present()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python backtest/run.py <pool_id>", file=sys.stderr)
        print(f"사용 가능한 pool_id: {sorted(BACKTEST_CONFIG.keys())}", file=sys.stderr)
        return 2

    pool_id = argv[1].strip().lower()
    run_backtest(pool_id, BACKTEST_CONFIG)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
