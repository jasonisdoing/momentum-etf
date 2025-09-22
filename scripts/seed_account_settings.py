"""Seed account-specific settings from existing country-level common_settings.

Usage examples:

    python scripts/seed_account_settings.py
    python scripts/seed_account_settings.py --mapping kor:m1,aus:a1,coin:c1 --dry-run
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

from utils.account_registry import get_account_info, load_accounts
from utils.db_manager import get_db_connection


DEFAULT_MAPPING = {
    "kor": "m1",
    "aus": "a1",
    "coin": "b1",
}


def parse_mapping(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"잘못된 매핑 형식입니다: '{pair}' (country:account)")
        country, account = [p.strip() for p in pair.split(":", 1)]
        if not country or not account:
            raise ValueError(f"잘못된 매핑 형식입니다: '{pair}'")
        mapping[country] = account
    if not mapping:
        raise ValueError("매핑이 비어 있습니다.")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed account_settings from common_settings")
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="country:account 리스트 (예: kor:m1,aus:a1,coin:c1)",
    )
    parser.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 예상 결과만 출력")
    args = parser.parse_args()

    mapping = DEFAULT_MAPPING if not args.mapping else parse_mapping(args.mapping)

    load_accounts(force_reload=True)
    db = get_db_connection()
    if db is None:
        raise SystemExit("MongoDB 연결에 실패했습니다.")

    account_settings = db.account_settings
    common_settings = db.common_settings

    print("[INFO] 시드 작업을 시작합니다.")
    for country, account in mapping.items():
        base_doc = common_settings.find_one({"country": country})
        if base_doc is None:
            print(f"[WARN] '{country}' 국가의 common_settings 문서를 찾을 수 없습니다. 건너뜁니다.")
            continue

        account_info = get_account_info(account)
        if account_info is None:
            print(f"[WARN] accounts.json 에 '{account}' 계좌가 없습니다. 건너뜁니다.")
            continue

        doc_to_save = dict(base_doc)
        doc_to_save.pop("_id", None)
        doc_to_save["account"] = account
        doc_to_save["country"] = country
        doc_to_save["migrated_from"] = f"common_settings::{country}"

        if args.dry_run:
            print(
                f"[DRY-RUN] account_settings upsert 예정: account={account}, keys={list(doc_to_save.keys())}"
            )
            continue

        result = account_settings.update_one(
            {"account": account},
            {
                "$set": {**doc_to_save, "updated_at": datetime.utcnow()},
                "$setOnInsert": {"created_at": datetime.utcnow()},
            },
            upsert=True,
        )

        action = (
            "updated" if result.modified_count else "inserted" if result.upserted_id else "noop"
        )
        print(f"[OK] account_settings {action}: account={account}, country={country}")

    print("[INFO] 시드 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()
