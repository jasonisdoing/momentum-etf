from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from bson import ObjectId
from pymongo import DESCENDING

from utils.db_manager import get_db_connection


def insert_trade_event(
    *,
    country: str,
    ticker: str,
    name: str | None = None,
    action: str,
    executed_at: datetime,
    memo: str | None,
    created_by: str,
    source: str = "streamlit",
) -> str:
    """`trades` 컬렉션에 매수/매도 이벤트를 기록합니다.

    Args:
        country: 국가 코드(kor, aus 등).
        ticker: 종목 코드/식별자.
        name: 종목명. 매수 시 필수 입력, 이후 수정 불가.
        action: "BUY" 또는 "SELL" 등 이벤트 타입.
        executed_at: 거래가 실행된 시각.
        memo: 추가 메모.
        created_by: 입력한 사용자 ID/이름.
        source: 입력 소스 구분값.

    Returns:
        MongoDB에 생성된 ObjectId 문자열.
    """

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    doc = {
        "country": country.strip().lower(),
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


def fetch_recent_trades(
    country: str | None = None, *, limit: int = 100, include_deleted: bool = False
) -> List[dict[str, Any]]:
    """최근 트레이드 목록을 반환합니다.

    Args:
        country: 국가 코드 (선택 사항)
        limit: 반환할 최대 항목 수
        include_deleted: 삭제된 항목 포함 여부
    """
    db = get_db_connection()
    if db is None:
        return []

    query: dict[str, Any] = {}
    if not include_deleted:
        query["deleted_at"] = {"$exists": False}
    else:
        query["deleted_at"] = {"$exists": True}

    if country:
        query["country"] = country.strip().lower()

    cursor = (
        db.trades.find(query)
        .sort([("executed_at", DESCENDING), ("_id", DESCENDING)])
        .limit(int(limit))
    )

    trades: List[dict[str, Any]] = []
    for doc in cursor:
        trades.append(
            {
                "id": str(doc.get("_id")),
                "country": str(doc.get("country") or ""),
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


def list_open_positions(country: str) -> List[dict[str, Any]]:
    """특정 국가의 최신 매수 상태(미매도) 종목 목록을 반환합니다."""

    country_norm = (country or "").strip().lower()
    if not country_norm:
        return []

    db = get_db_connection()
    if db is None:
        return []

    pipeline = [
        {
            "$match": {
                "country": country_norm,
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
    country: Optional[str] = None,
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
    if country is not None:
        fields["country"] = country.strip().lower()
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
        print(f"Error soft deleting trade {trade_id}: {e}")
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
        print(f"Error deleting trade {trade_id}: {e}")
        return False
