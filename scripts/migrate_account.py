#!/usr/bin/env python
"""계좌 ID 변경에 따른 저장 데이터를 마이그레이션한다."""

from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.cache_utils import _resolve_collection_name  # noqa: PLC2701
from utils.db_manager import get_db_connection


def _rename_account_in_portfolio_master(db, old_id: str, new_id: str) -> int:
    doc = db.portfolio_master.find_one({"master_id": "GLOBAL"})
    if not doc:
        return 0

    accounts = list(doc.get("accounts") or [])
    modified = 0
    for account in accounts:
        if str(account.get("account_id") or "").strip().lower() != old_id:
            continue
        account["account_id"] = new_id
        modified += 1

    if modified:
        db.portfolio_master.update_one({"master_id": "GLOBAL"}, {"$set": {"accounts": accounts}})
    return modified


def _rename_account_in_daily_snapshots(db, old_id: str, new_id: str) -> int:
    modified_docs = 0
    cursor = db.daily_snapshots.find({"accounts.account_id": old_id}, {"accounts": 1})
    for doc in cursor:
        accounts = list(doc.get("accounts") or [])
        changed = False
        for account in accounts:
            if str(account.get("account_id") or "").strip().lower() != old_id:
                continue
            account["account_id"] = new_id
            changed = True
        if not changed:
            continue
        db.daily_snapshots.update_one({"_id": doc["_id"]}, {"$set": {"accounts": accounts}})
        modified_docs += 1
    return modified_docs


def _rename_cache_collection(db, old_id: str, new_id: str) -> bool:
    old_collection = _resolve_collection_name(old_id)
    new_collection = _resolve_collection_name(new_id)
    existing = set(db.list_collection_names())

    if old_collection not in existing:
        return False
    if new_collection in existing:
        raise RuntimeError(f"새 캐시 컬렉션이 이미 존재합니다: {new_collection}. 수동 정리 후 다시 실행해야 합니다.")

    db[old_collection].rename(new_collection)
    return True


def migrate_account(old_id: str, new_id: str) -> None:
    old_norm = str(old_id or "").strip().lower()
    new_norm = str(new_id or "").strip().lower()
    if not old_norm or not new_norm:
        raise SystemExit("old_id와 new_id는 모두 필요합니다.")
    if old_norm == new_norm:
        raise SystemExit("old_id와 new_id가 동일합니다.")

    print(f"[{old_norm}] -> [{new_norm}] 마이그레이션을 시작합니다...")

    db = get_db_connection()
    if db is None:
        raise SystemExit("DB 연결 실패")

    stock_meta_result = db.stock_meta.update_many({"account_id": old_norm}, {"$set": {"account_id": new_norm}})
    print(f"  - `stock_meta`: {stock_meta_result.modified_count}건 변경")

    recommendation_result = db.stock_recommendations.update_many(
        {"account_id": old_norm},
        {"$set": {"account_id": new_norm}},
    )
    print(f"  - `stock_recommendations`: {recommendation_result.modified_count}건 변경")

    portfolio_master_count = _rename_account_in_portfolio_master(db, old_norm, new_norm)
    print(f"  - `portfolio_master.accounts`: {portfolio_master_count}건 변경")

    daily_snapshots_count = _rename_account_in_daily_snapshots(db, old_norm, new_norm)
    print(f"  - `daily_snapshots.accounts`: {daily_snapshots_count}건 변경")

    cache_renamed = _rename_cache_collection(db, old_norm, new_norm)
    print(f"  - `cache collection`: {'이름 변경 완료' if cache_renamed else '기존 컬렉션 없음'}")

    print("마이그레이션이 완료되었습니다.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="계좌 ID 변경에 따른 MongoDB 데이터를 마이그레이션합니다.")
    parser.add_argument("old_id", help="기존 계좌 ID")
    parser.add_argument("new_id", help="새 계좌 ID")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    migrate_account(args.old_id, args.new_id)
