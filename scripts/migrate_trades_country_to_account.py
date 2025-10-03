"""trades 컬렉션에서 `country` 필드를 `account`로 이전하는 마이그레이션 스크립트."""

from __future__ import annotations

from typing import Any

from utils.db_manager import get_db_connection


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def migrate() -> None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    trades = db.trades

    cursor = trades.find({"country": {"$exists": True}}, {"country": 1, "account": 1})

    migrated = 0
    skipped = 0

    for doc in cursor:
        doc_id = doc.get("_id")
        country_value = _normalize(doc.get("country"))
        account_value = _normalize(doc.get("account")) if "account" in doc else ""

        if account_value and account_value != country_value:
            print(
                f"[SKIP] _id={doc_id}: account='{account_value}' / country='{country_value}' 불일치로 건너뜁니다."
            )
            skipped += 1
            continue

        update_fields: dict[str, Any] = {}
        unset_fields: dict[str, int] = {}

        if country_value:
            if account_value != country_value:
                update_fields["account"] = country_value
            unset_fields["country"] = 1
        else:
            unset_fields["country"] = 1

        if not update_fields and not unset_fields:
            skipped += 1
            continue

        update_doc: dict[str, Any] = {}
        if update_fields:
            update_doc["$set"] = update_fields
        if unset_fields:
            update_doc["$unset"] = unset_fields

        result = trades.update_one({"_id": doc_id}, update_doc)
        if result.modified_count:
            migrated += 1
        else:
            skipped += 1

    print(f"완료: account 필드 갱신 {migrated}건, 변경 없음/건너뜀 {skipped}건")


if __name__ == "__main__":
    migrate()
