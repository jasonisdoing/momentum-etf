from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection
from utils.normalization import normalize_nullable_number, normalize_text

BUCKETS: dict[int, str] = {
    1: "1. 모멘텀",
    2: "2. 혁신기술",
    3: "3. 시장지수",
    4: "4. 배당방어",
    5: "5. 대체헷지",
}


def _format_deleted_date(value: Any) -> str:
    if not value:
        return "-"
    if isinstance(value, datetime):
        return value.isoformat()[:10]
    return str(value).strip()[:10] or "-"


def _load_accounts_payload() -> list[dict[str, Any]]:
    configs = load_account_configs()
    if not configs:
        raise RuntimeError("계좌 설정이 없습니다.")
    return [
        {
            "account_id": config["account_id"],
            "order": config["order"],
            "name": config["name"],
            "icon": config["icon"],
        }
        for config in configs
    ]


def _pick_account_id(accounts: list[dict[str, Any]], account_id: str | None) -> str:
    target = str(account_id or accounts[0]["account_id"]).strip().lower()
    if not target:
        raise RuntimeError("계좌를 찾을 수 없습니다.")
    return target


def load_active_stocks_table(account_id: str | None = None) -> dict[str, Any]:
    accounts = _load_accounts_payload()
    target_account_id = _pick_account_id(accounts, account_id)

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    docs = list(
        db.stock_meta.find(
            {
                "account_id": target_account_id,
                "is_deleted": {"$ne": True},
            },
            {
                "ticker": 1,
                "name": 1,
                "bucket": 1,
                "added_date": 1,
                "listing_date": 1,
                "1_week_avg_volume": 1,
                "1_week_earn_rate": 1,
                "2_week_earn_rate": 1,
                "1_month_earn_rate": 1,
                "3_month_earn_rate": 1,
                "6_month_earn_rate": 1,
                "12_month_earn_rate": 1,
            },
        )
    )

    rows = sorted(
        [
            {
                "ticker": normalize_text(doc.get("ticker"), ""),
                "name": normalize_text(doc.get("name"), ""),
                "bucket_id": int(doc.get("bucket") or 1),
                "bucket_name": BUCKETS.get(int(doc.get("bucket") or 1), BUCKETS[1]),
                "added_date": normalize_text(doc.get("added_date"), "-"),
                "listing_date": normalize_text(doc.get("listing_date"), "-"),
                "week_volume": normalize_nullable_number(doc.get("1_week_avg_volume")),
                "return_1w": normalize_nullable_number(doc.get("1_week_earn_rate")),
                "return_2w": normalize_nullable_number(doc.get("2_week_earn_rate")),
                "return_1m": normalize_nullable_number(doc.get("1_month_earn_rate")),
                "return_3m": normalize_nullable_number(doc.get("3_month_earn_rate")),
                "return_6m": normalize_nullable_number(doc.get("6_month_earn_rate")),
                "return_12m": normalize_nullable_number(doc.get("12_month_earn_rate")),
            }
            for doc in docs
        ],
        key=lambda row: (
            row["bucket_id"],
            -(row["return_1w"] if row["return_1w"] is not None else float("-inf")),
        ),
    )

    return {
        "accounts": accounts,
        "rows": rows,
        "account_id": target_account_id,
    }


def update_stock_bucket(account_id: str, ticker: str, bucket_id: int) -> None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    result = db.stock_meta.update_one(
        {
            "account_id": str(account_id or "").strip().lower(),
            "ticker": str(ticker or "").strip().upper(),
            "is_deleted": {"$ne": True},
        },
        {
            "$set": {
                "bucket": int(bucket_id),
                "updated_at": datetime.now(),
            }
        },
    )

    if result.matched_count == 0:
        raise RuntimeError("수정할 종목을 찾을 수 없습니다.")


def soft_delete_stock(account_id: str, ticker: str, reason: str | None = None) -> None:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    result = db.stock_meta.update_one(
        {
            "account_id": str(account_id or "").strip().lower(),
            "ticker": str(ticker or "").strip().upper(),
            "is_deleted": {"$ne": True},
        },
        {
            "$set": {
                "is_deleted": True,
                "deleted_reason": str(reason or "").strip(),
                "deleted_at": datetime.now(),
                "updated_at": datetime.now(),
            }
        },
    )

    if result.matched_count == 0:
        raise RuntimeError("삭제할 종목을 찾을 수 없습니다.")


def load_deleted_stocks_table(account_id: str | None = None) -> dict[str, Any]:
    accounts = _load_accounts_payload()
    target_account_id = _pick_account_id(accounts, account_id)

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    docs = list(
        db.stock_meta.find(
            {
                "account_id": target_account_id,
                "is_deleted": True,
            },
            {
                "ticker": 1,
                "name": 1,
                "bucket": 1,
                "listing_date": 1,
                "1_week_avg_volume": 1,
                "1_week_earn_rate": 1,
                "2_week_earn_rate": 1,
                "1_month_earn_rate": 1,
                "3_month_earn_rate": 1,
                "6_month_earn_rate": 1,
                "12_month_earn_rate": 1,
                "deleted_at": 1,
                "deleted_reason": 1,
            },
        )
    )

    rows = sorted(
        [
            {
                "ticker": normalize_text(doc.get("ticker"), ""),
                "name": normalize_text(doc.get("name"), ""),
                "bucket_id": int(doc.get("bucket") or 1),
                "bucket_name": BUCKETS.get(int(doc.get("bucket") or 1), BUCKETS[1]),
                "listing_date": normalize_text(doc.get("listing_date"), "-"),
                "week_volume": normalize_nullable_number(doc.get("1_week_avg_volume")),
                "return_1w": normalize_nullable_number(doc.get("1_week_earn_rate")),
                "return_2w": normalize_nullable_number(doc.get("2_week_earn_rate")),
                "return_1m": normalize_nullable_number(doc.get("1_month_earn_rate")),
                "return_3m": normalize_nullable_number(doc.get("3_month_earn_rate")),
                "return_6m": normalize_nullable_number(doc.get("6_month_earn_rate")),
                "return_12m": normalize_nullable_number(doc.get("12_month_earn_rate")),
                "deleted_date": _format_deleted_date(doc.get("deleted_at")),
                "deleted_reason": normalize_text(doc.get("deleted_reason"), "-"),
            }
            for doc in docs
        ],
        key=lambda row: (row["bucket_id"], row["deleted_date"]),
        reverse=True,
    )

    return {
        "accounts": accounts,
        "rows": rows,
        "account_id": target_account_id,
    }


def restore_deleted_stocks(account_id: str, tickers: list[str]) -> int:
    account_norm = str(account_id or "").strip().lower()
    ticker_list = [str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()]
    if not account_norm or not ticker_list:
        raise RuntimeError("복구할 종목을 선택하세요.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    now = datetime.now()
    result = db.stock_meta.update_many(
        {
            "account_id": account_norm,
            "ticker": {"$in": ticker_list},
            "is_deleted": True,
        },
        {
            "$set": {
                "is_deleted": False,
                "deleted_at": None,
                "deleted_reason": None,
                "added_date": now.date().isoformat(),
                "updated_at": now,
            }
        },
    )
    return int(result.modified_count)


def hard_delete_stocks(account_id: str, tickers: list[str]) -> int:
    account_norm = str(account_id or "").strip().lower()
    ticker_list = [str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()]
    if not account_norm or not ticker_list:
        raise RuntimeError("삭제할 종목을 선택하세요.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    result = db.stock_meta.delete_many(
        {
            "account_id": account_norm,
            "ticker": {"$in": ticker_list},
            "is_deleted": True,
        }
    )
    return int(result.deleted_count)
