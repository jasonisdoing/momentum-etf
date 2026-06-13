"""pools.json 의 편집 가능한 5개 값을 pool_settings 컬렉션에 시드한다.

DB 가 5개 값(TOP_N_HOLD/HOLDING_BONUS_SCORE/MA_TYPE/MA_MONTHS/RSI_LIMIT)의 단일 소스이므로,
신규 환경/신규 풀 추가 후 한 번 실행해 DB 를 채운다. 기본은 빠진 것만 채우고(기존값 보존),
--overwrite 시 pools.json 값으로 전부 덮어쓴다.

    python scripts/seed_pool_settings.py            # 빠진 것만 시드 (안전)
    python scripts/seed_pool_settings.py --overwrite # pools.json 값으로 전부 덮어쓰기
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import load_env_if_present
from utils.logger import get_app_logger
from utils.pool_settings_store import seed_from_pools_json

logger = get_app_logger()


def main() -> None:
    load_env_if_present()
    parser = argparse.ArgumentParser(description="pool_settings 시드")
    parser.add_argument("--overwrite", action="store_true", help="pools.json 값으로 전부 덮어쓰기")
    args = parser.parse_args()

    summary = seed_from_pools_json(overwrite=args.overwrite)
    logger.info(
        "[seed_pool_settings] 완료 — 신규 %d / 덮어씀 %d / 유지 %d",
        len(summary["seeded"]),
        len(summary["overwritten"]),
        len(summary["skipped"]),
    )
    logger.info("[seed_pool_settings] 신규: %s", summary["seeded"])
    logger.info("[seed_pool_settings] 덮어씀: %s", summary["overwritten"])
    logger.info("[seed_pool_settings] 유지: %s", summary["skipped"])


if __name__ == "__main__":
    main()
