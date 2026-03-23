#!/usr/bin/env python
"""pool 종목을 account 종목으로 이관하고 pool 종목 문서를 제거합니다."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from pymongo import UpdateOne

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.cache_utils import _resolve_collection_name
from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

POOL_TO_ACCOUNT = {
    "kor": "kor_account",
    "tax": "pension_account",
    "core": "core_account",
    "aus": "aus_account",
}


def main() -> int:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_meta 마이그레이션을 진행할 수 없습니다.")

    coll = db["stock_meta"]
    now = datetime.now(timezone.utc)

    for pool_id, account_id in POOL_TO_ACCOUNT.items():
        docs = list(coll.find({"account_id": pool_id}))
        logger.info("[%s -> %s] 이관 대상 %d건", pool_id, account_id, len(docs))
        if docs:
            operations: list[UpdateOne] = []
            for doc in docs:
                ticker = str(doc.get("ticker") or "").strip().upper()
                if not ticker:
                    continue

                new_doc = {k: v for k, v in doc.items() if k != "_id"}
                original_created_at = new_doc.pop("created_at", now)
                new_doc["account_id"] = account_id
                new_doc["ticker"] = ticker
                new_doc["updated_at"] = now

                operations.append(
                    UpdateOne(
                        {"account_id": account_id, "ticker": ticker},
                        {"$set": new_doc, "$setOnInsert": {"created_at": original_created_at}},
                        upsert=True,
                    )
                )

            if operations:
                coll.bulk_write(operations, ordered=False)

            delete_result = coll.delete_many({"account_id": pool_id})
            logger.info("[%s] 기존 pool 문서 삭제 %d건", pool_id, delete_result.deleted_count)

        source_cache_name = _resolve_collection_name(pool_id)
        target_cache_name = _resolve_collection_name(account_id)
        source_cache = db[source_cache_name]
        target_cache = db[target_cache_name]
        cache_docs = list(source_cache.find({}, {"_id": 0}))
        logger.info("[%s -> %s] 캐시 이관 대상 %d건", source_cache_name, target_cache_name, len(cache_docs))
        if cache_docs:
            target_cache.drop()
            target_cache.insert_many(cache_docs, ordered=False)
            try:
                target_cache.create_index("ticker", unique=True, name="ticker_unique", background=True)
            except Exception:
                pass
        source_cache.drop()
        logger.info("[%s] 기존 pool 캐시 컬렉션 삭제 완료", source_cache_name)

    logger.info("pool -> account stock_meta 마이그레이션 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
