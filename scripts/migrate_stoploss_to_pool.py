"""손절 기준(%)을 계좌 설정 → 종목풀 설정으로 옮기는 일회성 마이그레이션.

- `pool_settings` 전 문서에 ``STOPLOSS_THRESHOLD_PCT = -10.0`` 을 넣는다(이미 있으면 유지).
- `account_settings` 전 문서에서 ``stoploss_threshold_pct`` 필드를 제거한다.

실행 전 원본을 scratch 백업 파일로 남긴다. 실행 후 이 스크립트는 삭제한다.

    .venv/bin/python -m scripts.migrate_stoploss_to_pool
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.db_manager import get_db_connection

DEFAULT_STOPLOSS_PCT = -10.0
BACKUP_DIR = Path("backups")


def main() -> None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패")

    pools = list(db["pool_settings"].find({}))
    accounts = list(db["account_settings"].find({}))

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"stoploss_migration_{stamp}.json"
    backup_path.write_text(
        json.dumps({"pool_settings": pools, "account_settings": accounts}, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    print(f"백업 저장: {backup_path}")

    added = 0
    for doc in pools:
        pool_id = str(doc.get("_id") or "")
        if not pool_id or pool_id.startswith("__"):
            continue
        if doc.get("STOPLOSS_THRESHOLD_PCT") is not None:
            print(f"  건너뜀(이미 설정됨) {pool_id}: {doc['STOPLOSS_THRESHOLD_PCT']}%")
            continue
        db["pool_settings"].update_one({"_id": doc["_id"]}, {"$set": {"STOPLOSS_THRESHOLD_PCT": DEFAULT_STOPLOSS_PCT}})
        added += 1
        print(f"  설정 {pool_id}: {DEFAULT_STOPLOSS_PCT}%")

    removed = (
        db["account_settings"]
        .update_many(
            {"stoploss_threshold_pct": {"$exists": True}},
            {"$unset": {"stoploss_threshold_pct": ""}},
        )
        .modified_count
    )

    print(f"완료 — 종목풀 {added}개 손절 기준 설정, 계좌 {removed}개 문서에서 stoploss_threshold_pct 제거")


if __name__ == "__main__":
    main()
