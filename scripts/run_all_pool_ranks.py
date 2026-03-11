"""모든 종목풀 랭킹을 순차 실행합니다."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.identifier_guard import ensure_account_pool_id_separation
from utils.logger import get_app_logger
from utils.pool_registry import list_available_pools

logger = get_app_logger()


def main() -> int:
    load_env_if_present()
    ensure_account_pool_id_separation()

    pool_ids = list_available_pools()
    if not pool_ids:
        logger.error("실행할 종목풀이 없습니다.")
        return 1

    logger.info("전체 종목풀 랭킹 실행 시작: %s", pool_ids)

    failed: list[str] = []
    for index, pool_id in enumerate(pool_ids, start=1):
        logger.info(" -> 종목풀 랭킹 실행 중: %d/%d - %s", index, len(pool_ids), pool_id)
        result = subprocess.run([sys.executable, "rank.py", pool_id], cwd=str(ROOT_DIR))
        if result.returncode != 0:
            failed.append(pool_id)

    if failed:
        logger.error("실패한 종목풀: %s", ", ".join(failed))
        return 1

    logger.info("전체 종목풀 랭킹 실행 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
