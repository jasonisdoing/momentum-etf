from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from bson import ObjectId

from utils.db_manager import get_db_connection
from utils.logger import APP_LABEL, get_app_logger

logger = get_app_logger()
COLLECTION_NAME = "pool_rankings"

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _normalize_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _get_collection():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")
    return db[COLLECTION_NAME]


def _now_kst() -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Seoul"))
        except Exception:
            pass
    return datetime.utcnow()


def save_pool_rank_payload(
    *,
    pool_id: str,
    country_code: str,
    rows: list[dict[str, Any]],
    base_date: Any | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pool_norm = (pool_id or "").strip().lower()
    if not pool_norm:
        raise RuntimeError("pool_id is required")

    coll = _get_collection()
    now = _now_kst()
    base_dt = _normalize_datetime(base_date)

    update_doc = {
        "pool_id": pool_norm,
        "country_code": (country_code or "").strip().lower(),
        "rows": rows,
        "config": config or {},
        "base_date": base_dt,
        "updated_at": now,
        "updated_by": APP_LABEL,
    }

    result = coll.update_one(
        {"pool_id": pool_norm},
        {"$set": update_doc, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )

    if result.upserted_id is not None:
        doc_id = result.upserted_id
    else:
        existing = coll.find_one({"pool_id": pool_norm}, {"_id": 1})
        doc_id = existing.get("_id") if existing else None

    logger.info("MongoDB에 풀 랭킹을 저장했습니다 (pool=%s, count=%d)", pool_norm.upper(), len(rows))

    return {
        "pool_id": pool_norm,
        "document_id": str(doc_id) if isinstance(doc_id, ObjectId) else doc_id,
        "updated_at": now,
    }


def fetch_latest_pool_rank(pool_id: str) -> dict[str, Any] | None:
    pool_norm = (pool_id or "").strip().lower()
    if not pool_norm:
        return None

    coll = _get_collection()
    return coll.find_one({"pool_id": pool_norm}, sort=[("base_date", -1), ("updated_at", -1)])
