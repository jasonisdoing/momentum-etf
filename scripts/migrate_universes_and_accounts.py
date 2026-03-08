#!/usr/bin/env python
"""종목군/계좌 ID 분리를 위한 데이터 마이그레이션 도구.

기본 동작(dry-run): 변경 없이 계획/검증만 출력
실행 모드(--execute): stock_meta/추천/포트폴리오 계좌 ID를 실제로 복제
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pymongo import UpdateOne

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection
from utils.env import load_env_if_present

# 요청하신 종목군 매핑 (idempotent를 위해 소스 후보를 순서대로 시도)
POOL_SOURCE_CANDIDATES: dict[str, list[str]] = {
    "kor_kr": ["kor_kr", "kor_account"],  # 국내상장 국내 ETF
    "kor_us": ["kor_pension", "pension_account"],  # 국내상장 해외 ETF
    "us": ["us", "us_account"],  # 미국 ETF
    "aus": ["aus", "aus_account"],  # 호주 ETF
}

# 새 WEIGHT 계좌 매핑 (재실행 시 현재 계좌를 우선 소스로 사용)
NEW_ACCOUNT_SOURCE_CANDIDATES: dict[str, list[str]] = {
    "kor_account": ["kor_account", "kor_kr"],
    "pension_account": ["pension_account", "kor_pension"],
    "kor_save_account": ["kor_save_account", "kor_us"],
    "us_account": ["us_account", "us"],
    "aus_account": ["aus_account", "aus"],
}

# 더 이상 계좌로 쓰지 않는 기존 ID (정리 옵션용)
RETIRED_ACCOUNT_IDS = ["kor_pension", "kor_us"]


@dataclass
class CopyStats:
    source: str
    target: str
    total_src: int
    active_src: int
    total_tgt_after: int
    active_tgt_after: int


def _count(coll, account_id: str) -> tuple[int, int]:
    total = coll.count_documents({"account_id": account_id})
    active = coll.count_documents({"account_id": account_id, "is_deleted": {"$ne": True}})
    return total, active


def _copy_stock_meta(coll, source: str, target: str, execute: bool) -> CopyStats:
    src_docs = list(coll.find({"account_id": source}, {"_id": 0}))
    total_src = len(src_docs)
    active_src = sum(1 for doc in src_docs if doc.get("is_deleted") is not True)

    if execute and src_docs:
        now = datetime.now(timezone.utc)
        ops: list[UpdateOne] = []
        for doc in src_docs:
            ticker = str(doc.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            payload = dict(doc)
            payload.pop("created_at", None)
            payload["account_id"] = target
            payload["ticker"] = ticker
            payload["updated_at"] = now
            ops.append(
                UpdateOne(
                    {"account_id": target, "ticker": ticker},
                    {"$set": payload, "$setOnInsert": {"created_at": now}},
                    upsert=True,
                )
            )
        if ops:
            coll.bulk_write(ops, ordered=False)

    total_tgt_after, active_tgt_after = _count(coll, target)
    return CopyStats(
        source=source,
        target=target,
        total_src=total_src,
        active_src=active_src,
        total_tgt_after=total_tgt_after,
        active_tgt_after=active_tgt_after,
    )


def _copy_recommendation_doc(coll, source: str, target: str, execute: bool) -> bool:
    doc = coll.find_one({"account_id": source}, {"_id": 0})
    if not doc:
        return False
    if execute:
        now = datetime.now(timezone.utc)
        doc["account_id"] = target
        doc.pop("created_at", None)
        doc["updated_at"] = now
        coll.update_one(
            {"account_id": target},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
    return True


def _copy_portfolio_master_accounts(coll, mapping: dict[str, str], execute: bool) -> int:
    doc = coll.find_one({"master_id": "GLOBAL"})
    if not doc:
        return 0

    accounts = list(doc.get("accounts") or [])
    if not isinstance(accounts, list):
        return 0

    by_id: dict[str, dict[str, Any]] = {}
    for item in accounts:
        if isinstance(item, dict) and item.get("account_id"):
            by_id[str(item["account_id"]).strip().lower()] = dict(item)

    copied = 0
    for new_id, source in mapping.items():
        src = source.strip().lower()
        dst = new_id.strip().lower()
        src_entry = by_id.get(src)
        if not src_entry:
            continue
        if dst in by_id:
            continue
        cloned = dict(src_entry)
        cloned["account_id"] = dst
        by_id[dst] = cloned
        copied += 1

    if execute and copied:
        coll.update_one(
            {"master_id": "GLOBAL"},
            {"$set": {"accounts": list(by_id.values())}},
            upsert=True,
        )
    return copied


def _cleanup_retired_stock_meta(coll, execute: bool) -> int:
    protected_ids = set(POOL_SOURCE_CANDIDATES.keys())
    cleanup_targets = [acc for acc in RETIRED_ACCOUNT_IDS if acc not in protected_ids]
    if not cleanup_targets:
        return 0
    if not execute:
        return coll.count_documents({"account_id": {"$in": cleanup_targets}})
    result = coll.delete_many({"account_id": {"$in": cleanup_targets}})
    return int(result.deleted_count)


def _print_stats(title: str, rows: list[CopyStats]) -> None:
    print(f"\n[{title}]")
    if not rows:
        print("  - 작업 없음")
        return
    for row in rows:
        print(
            f"  - {row.source} -> {row.target}: "
            f"src(total={row.total_src}, active={row.active_src}) | "
            f"target_after(total={row.total_tgt_after}, active={row.active_tgt_after})"
        )


def _pick_source(coll, candidates: list[str]) -> str:
    for candidate in candidates:
        if coll.count_documents({"account_id": candidate}) > 0:
            return candidate
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="종목군/계좌 분리 마이그레이션")
    parser.add_argument("--execute", action="store_true", help="실제 DB 쓰기 수행")
    parser.add_argument(
        "--cleanup-retired",
        action="store_true",
        help="retired 계좌 ID(kor_pension, kor_us) stock_meta 삭제",
    )
    args = parser.parse_args()

    load_env_if_present()
    db = get_db_connection()
    if db is None:
        print("[오류] DB 연결 실패: MONGO_DB_CONNECTION_STRING 확인 필요")
        return 1

    stock_meta = db["stock_meta"]
    rec_coll = db["stock_recommendations"]
    portfolio_master = db["portfolio_master"]

    print("[모드]", "EXECUTE" if args.execute else "DRY-RUN")

    account_rows: list[CopyStats] = []
    account_source_map: dict[str, str] = {}
    for new_account, candidates in NEW_ACCOUNT_SOURCE_CANDIDATES.items():
        source = _pick_source(stock_meta, candidates)
        account_source_map[new_account] = source
        account_rows.append(_copy_stock_meta(stock_meta, source, new_account, args.execute))

    pool_rows: list[CopyStats] = []
    for target_pool, candidates in POOL_SOURCE_CANDIDATES.items():
        source = _pick_source(stock_meta, candidates)
        pool_rows.append(_copy_stock_meta(stock_meta, source, target_pool, args.execute))

    rec_hits = 0
    for new_account, source in account_source_map.items():
        if _copy_recommendation_doc(rec_coll, source, new_account, args.execute):
            rec_hits += 1

    copied_master = _copy_portfolio_master_accounts(portfolio_master, account_source_map, args.execute)

    _print_stats("Pool Copy (RANK 종목군)", pool_rows)
    _print_stats("Account Copy (신규 WEIGHT 계좌)", account_rows)

    print("\n[추천/포트폴리오]")
    print(f"  - stock_recommendations 복제 대상 발견: {rec_hits}건")
    print(f"  - portfolio_master GLOBAL 계좌 복제: {copied_master}건")

    if args.cleanup_retired:
        deleted = _cleanup_retired_stock_meta(stock_meta, args.execute)
        print(f"\n[정리] retired stock_meta 도큐먼트: {deleted}건 {'삭제' if args.execute else '삭제 예정'}")

    print("\n완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
