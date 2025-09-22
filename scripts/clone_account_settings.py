"""Clone existing account_settings documents to new accounts.

Usage examples:
    python scripts/clone_account_settings.py --source-account m1 --target-accounts m2 m3
    python scripts/clone_account_settings.py --source-account m1 --target-accounts m2 m3 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_manager import get_db_connection


def clone_account_settings(
    source_account: str, target_accounts: List[str], dry_run: bool = False
) -> None:
    db = get_db_connection()
    if db is None:
        raise SystemExit("MongoDB 연결에 실패했습니다.")

    col = db.account_settings
    source_doc = col.find_one({"account": source_account})
    if not source_doc:
        raise SystemExit(f"계좌 '{source_account}'에 대한 account_settings 문서를 찾을 수 없습니다.")

    source_doc.pop("_id", None)

    for target in target_accounts:
        if target == source_account:
            print(f"[SKIP] 대상 계좌 '{target}'가 원본과 동일합니다.")
            continue

        doc_to_save = dict(source_doc)
        doc_to_save.pop("created_at", None)
        doc_to_save["account"] = target
        doc_to_save["migrated_from"] = source_account
        doc_to_save["updated_at"] = datetime.now(timezone.utc)

        if dry_run:
            print(f"[DRY-RUN] account_settings upsert 예정: account={target}")
            continue

        result = col.update_one(
            {"account": target},
            {
                "$set": doc_to_save,
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )

        action = (
            "updated" if result.modified_count else "inserted" if result.upserted_id else "noop"
        )
        print(f"[OK] account_settings {action}: account={target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone account settings to new accounts")
    parser.add_argument("--source-account", required=True, help="원본 계좌")
    parser.add_argument(
        "--target-accounts",
        nargs="+",
        required=True,
        help="복사될 대상 계좌들",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB에 쓰지 않고 작업 내용을 출력만 합니다.",
    )
    args = parser.parse_args()

    clone_account_settings(
        source_account=args.source_account,
        target_accounts=args.target_accounts,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
