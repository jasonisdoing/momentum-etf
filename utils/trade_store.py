from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from bson import ObjectId
from pymongo import DESCENDING

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()


def insert_trade_event(
    *,
    account_id: str,
    ticker: str,
    name: str | None = None,
    action: str,
    executed_at: datetime,
    memo: str | None,
    created_by: str,
    source: str = "streamlit",
    country_code: str | None = None,
) -> str:
    """`trades` 컬렉션에 매수/매도 이벤트를 기록합니다."""

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    account_norm = (account_id or "").strip().lower()
    country_norm = (country_code or account_norm).strip().lower()
    if not account_norm:
        raise ValueError("계정 ID를 지정해야 합니다.")

    doc = {
        "account": account_norm,
        "country_code": country_norm,
        "ticker": ticker,
        "action": action.upper(),
        "name": (name or "").strip(),
        "executed_at": executed_at,
        "memo": memo or "",
        "created_by": created_by,
        "source": source,
        "created_at": datetime.utcnow(),
    }

    if not doc["name"] and action.upper() == "SELL":
        doc.pop("name")

    result = db.trades.insert_one(doc)
    return str(result.inserted_id)


def migrate_account_id(old_account_id: str, new_account_id: str) -> dict[str, int]:
    """`trades` 컬렉션에서 계정 ID를 일괄 변경합니다."""

    old_norm = (old_account_id or "").strip().lower()
    new_norm = (new_account_id or "").strip().lower()
    if not old_norm or not new_norm:
        raise ValueError("old_account_id와 new_account_id는 비어 있을 수 없습니다.")
    if old_norm == new_norm:
        return {"matched": 0, "modified": 0}

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    result_account = db.trades.update_many(
        {"account": old_norm},
        {"$set": {"account": new_norm}},
    )

    return {
        "matched": int(result_account.matched_count),
        "modified": int(result_account.modified_count),
    }


def delete_account_trades(account_id: str) -> dict[str, int]:
    """지정한 계정 ID의 거래 이력을 모두 삭제합니다."""

    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id는 비어 있을 수 없습니다.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    result_account = db.trades.delete_many({"account": account_norm})
    deleted_count = int(result_account.deleted_count)

    logger.info(
        "trades 컬렉션에서 계정 '%s' 데이터를 삭제했습니다. deleted=%d",
        account_norm,
        deleted_count,
    )

    return {
        "deleted": deleted_count,
    }


def fetch_recent_trades(account_id: str | None = None, *, limit: int = 100, include_deleted: bool = False) -> List[dict[str, Any]]:
    """최근 트레이드 목록을 반환합니다."""
    db = get_db_connection()
    if db is None:
        return []

    query: dict[str, Any] = {}
    if not include_deleted:
        query["deleted_at"] = {"$exists": False}
    else:
        query["deleted_at"] = {"$exists": True}

    if account_id:
        query["account"] = account_id.strip().lower()

    cursor = db.trades.find(query).sort([("executed_at", DESCENDING), ("_id", DESCENDING)]).limit(int(limit))

    trades: List[dict[str, Any]] = []
    for doc in cursor:
        trades.append(
            {
                "id": str(doc.get("_id")),
                "account": str(doc.get("account") or ""),
                "country_code": str(doc.get("country_code") or ""),
                "ticker": str(doc.get("ticker") or ""),
                "action": str(doc.get("action") or ""),
                "name": str(doc.get("name") or ""),
                "executed_at": doc.get("executed_at"),
                "memo": doc.get("memo", ""),
                "created_by": doc.get("created_by"),
                "deleted_at": doc.get("deleted_at"),
                "is_deleted": "deleted_at" in doc,
            }
        )
    return trades


def list_open_positions(account_id: str) -> List[dict[str, Any]]:
    """특정 계정의 최신 매수 상태(미매도) 종목 목록을 반환합니다."""

    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        return []

    db = get_db_connection()
    if db is None:
        return []

    pipeline = [
        {
            "$match": {
                "account": account_norm,
                "ticker": {"$ne": None},
                "deleted_at": {"$exists": False},
            }
        },
        {"$sort": {"executed_at": 1, "_id": 1}},
        {
            "$group": {
                "_id": "$ticker",
                "last_action": {"$last": "$action"},
                "last_doc": {"$last": "$$ROOT"},
            }
        },
        {"$match": {"last_action": "BUY"}},
        {"$sort": {"_id": 1}},
    ]

    try:
        results = list(db.trades.aggregate(pipeline))
    except Exception:
        return []

    holdings: List[dict[str, Any]] = []
    for item in results:
        last_doc = item.get("last_doc") or {}
        holdings.append(
            {
                "id": str(last_doc.get("_id")) if last_doc.get("_id") else "",
                "ticker": str(item.get("_id") or "").upper(),
                "last_action": item.get("last_action"),
                "executed_at": last_doc.get("executed_at"),
                "name": str(last_doc.get("name", "")),
                "memo": last_doc.get("memo", ""),
            }
        )
    return holdings


def update_trade_event(
    trade_id: str,
    *,
    account_id: Optional[str] = None,
    ticker: Optional[str] = None,
    action: Optional[str] = None,
    executed_at: Optional[datetime] = None,
    memo: Optional[str] = None,
) -> bool:
    """트레이드 문서를 업데이트합니다."""

    trade_id = (trade_id or "").strip()
    if not trade_id:
        return False

    db = get_db_connection()
    if db is None:
        return False

    try:
        object_id = ObjectId(trade_id)
    except Exception:
        return False

    fields: dict[str, Any] = {}
    if account_id is not None:
        fields["account"] = account_id.strip().lower()
    if ticker is not None:
        fields["ticker"] = ticker.strip()
    if action is not None:
        fields["action"] = action.upper()
    if executed_at is not None:
        fields["executed_at"] = executed_at
    if memo is not None:
        fields["memo"] = memo

    if not fields:
        return False

    result = db.trades.update_one({"_id": object_id}, {"$set": fields})
    return result.modified_count > 0


def soft_delete_trade(trade_id: str) -> bool:
    """트레이드 문서를 소프트 삭제합니다."""
    db = get_db_connection()
    if not db:
        return False

    try:
        result = db.trades.update_one(
            {"_id": ObjectId(trade_id)},
            {"$set": {"deleted_at": datetime.utcnow()}},
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error("소프트 삭제 실패 (trade_id=%s): %s", trade_id, e)
        return False


def delete_trade(trade_id: str) -> bool:
    """트레이드 문서를 완전히 삭제합니다.

    Args:
        trade_id: 삭제할 트레이드의 ID

    Returns:
        삭제 성공 여부 (True/False)
    """
    db = get_db_connection()
    if db is None:
        return False

    try:
        result = db.trades.delete_one({"_id": ObjectId(trade_id)})
        return result.deleted_count > 0
    except Exception as e:
        logger.error("거래 삭제 실패 (trade_id=%s): %s", trade_id, e)
        return False
