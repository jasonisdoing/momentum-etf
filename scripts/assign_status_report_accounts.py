"""Assign account codes to existing status_reports documents.

Usage:
    python scripts/assign_status_report_accounts.py
    python scripts/assign_status_report_accounts.py --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_manager import get_db_connection


COUNTRY_TO_ACCOUNT: Dict[str, str] = {
    "kor": "m1",
    "aus": "a1",
    "coin": "b1",
}


def assign_accounts(dry_run: bool = False) -> None:
    db = get_db_connection()
    if db is None:
        raise SystemExit("MongoDB 연결에 실패했습니다.")

    col = db.status_reports
    total_updates = 0

    for country, account in COUNTRY_TO_ACCOUNT.items():
        query = {
            "country": country,
            "$or": [
                {"account": {"$exists": False}},
                {"account": None},
                {"account": ""},
            ],
        }

        docs = list(col.find(query, {"_id": 1}))
        if not docs:
            print(f"[{country}] 업데이트할 문서가 없습니다.")
            continue

        ids = [doc["_id"] for doc in docs]
        print(f"[{country}] {len(ids)}건에 account='{account}' 적용")
        if dry_run:
            continue

        result = col.update_many(
            {"_id": {"$in": ids}},
            {
                "$set": {
                    "account": account,
                    "updated_at": datetime.utcnow(),
                }
            },
        )
        total_updates += result.modified_count

    if dry_run:
        print("[DRY-RUN] 실제 업데이트는 수행하지 않았습니다.")
    else:
        print(f"[DONE] account 필드를 갱신한 문서 수: {total_updates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign account to status_reports")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB를 수정하지 않고 예상 결과만 출력",
    )
    args = parser.parse_args()

    assign_accounts(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
