#!/usr/bin/env python
"""기존 stocks.json 파일을 MongoDB stock_meta 컬렉션으로 마이그레이션합니다."""

from __future__ import annotations

import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import load_env_if_present
from utils.logger import get_app_logger
from utils.settings_loader import list_available_accounts
from utils.stock_list_io import load_stocks_from_file, save_etfs

logger = get_app_logger()


def migrate_all() -> None:
    """모든 계좌의 stocks.json을 DB로 마이그레이션합니다."""
    load_env_if_present()

    accounts = list_available_accounts()
    if not accounts:
        logger.error("사용 가능한 계좌가 없습니다.")
        return

    total_migrated = 0

    for account_id in sorted(accounts):
        data = load_stocks_from_file(account_id)
        if not data:
            logger.warning("[%s] stocks.json이 비어있거나 없습니다. 건너뜁니다.", account_id)
            continue

        logger.info("[%s] %d개 종목을 DB로 마이그레이션합니다...", account_id, len(data))

        try:
            save_etfs(account_id, data)
            total_migrated += len(data)
            logger.info("[%s] ✅ 마이그레이션 완료.", account_id)
        except Exception as exc:
            logger.error("[%s] ❌ 마이그레이션 실패: %s", account_id, exc)

    logger.info("=== 전체 마이그레이션 완료: %d개 종목 ===", total_migrated)

    # 검증
    _verify(accounts)


def _verify(accounts: list[str]) -> None:
    """DB와 파일 간 건수를 비교하여 검증합니다."""
    from utils.stock_list_io import get_all_etfs

    logger.info("--- 검증 시작 ---")
    all_ok = True

    for account_id in sorted(accounts):
        file_data = load_stocks_from_file(account_id)
        file_count = len(file_data)

        # 캐시를 무효화하여 DB에서 새로 읽기
        from utils.stock_list_io import _invalidate_cache

        _invalidate_cache(account_id)

        db_data = get_all_etfs(account_id)
        db_count = len(db_data)

        status = "✅" if file_count == db_count else "❌"
        if file_count != db_count:
            all_ok = False
        logger.info("[%s] %s 파일: %d개, DB: %d개", account_id, status, file_count, db_count)

    if all_ok:
        logger.info("--- 검증 결과: 모두 일치합니다. ---")
    else:
        logger.warning("--- 검증 결과: 불일치가 있습니다. 확인이 필요합니다. ---")


if __name__ == "__main__":
    migrate_all()
