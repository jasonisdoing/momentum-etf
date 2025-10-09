from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId

import pandas as pd

try:  # pragma: no cover - numpy is optional at runtime
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = get_app_logger()

COLLECTION_NAME = "stock_recommendations"


def _make_json_safe(obj: Any) -> Any:
    """Convert arbitrary Python objects into Mongo serializable structures."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (datetime, pd.Timestamp)):
        try:
            return obj.to_pydatetime()  # type: ignore[attr-defined]
        except AttributeError:
            return obj

    if np is not None and isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, pd.Series):
        return [_make_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [{k: _make_json_safe(v) for k, v in record.items()} for record in obj.to_dict(orient="records")]

    return str(obj)


def _normalize_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        try:
            return value.to_pydatetime()
        except Exception:
            return value.to_pydatetime()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _now_kst() -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Seoul"))
        except Exception:
            pass
    return datetime.utcnow()


def _get_collection():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")
    return db[COLLECTION_NAME]


def save_recommendation_payload(
    payload: Any,
    *,
    account_id: str,
    country_code: Optional[str] = None,
    base_date: Optional[Any] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist recommendation payload to MongoDB and return metadata."""

    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise RuntimeError("account_id is required to save recommendations")

    collection = _get_collection()

    safe_payload = _make_json_safe(payload)
    safe_summary = _make_json_safe(summary) if summary is not None else None
    base_datetime = _normalize_datetime(base_date)
    now = _now_kst()

    update_doc: Dict[str, Any] = {
        "account_id": account_norm,
        "country_code": (country_code or "").strip().lower() or None,
        "recommendations": safe_payload,
        "summary": safe_summary,
        "base_date": base_datetime,
        "updated_at": now,
    }

    update_operations: Dict[str, Any] = {
        "$set": update_doc,
        "$setOnInsert": {"created_at": now},
    }

    result = collection.update_one({"account_id": account_norm}, update_operations, upsert=True)

    if result.upserted_id is not None:
        doc_id = result.upserted_id
    else:
        existing = collection.find_one({"account_id": account_norm}, {"_id": 1})
        doc_id = existing["_id"] if existing else None

    logger.info(
        "MongoDB에 추천 결과를 저장했습니다 (account=%s, document_id=%s, count=%s)",
        account_norm.upper(),
        doc_id,
        len(safe_payload) if isinstance(safe_payload, list) else "?",
    )

    return {
        "account_id": account_norm,
        "document_id": str(doc_id) if isinstance(doc_id, ObjectId) else doc_id,
        "updated_at": now,
    }


def save_recommendation_report(
    report: Any,
    *,
    results_dir: Any | None = None,  # 유지: 기존 시그니처 호환
) -> Dict[str, Any]:
    """Persist a RecommendationReport-like object and return Mongo metadata."""

    account_id = getattr(report, "account_id", "")
    recommendations = getattr(report, "recommendations", None)

    if recommendations is None:
        raise RuntimeError("Recommendation report must include recommendations data")

    country_code = getattr(report, "country_code", None)
    base_date = getattr(report, "base_date", None)
    summary = getattr(report, "summary_data", None)

    return save_recommendation_payload(
        recommendations,
        account_id=str(account_id),
        country_code=country_code,
        base_date=base_date,
        summary=summary,
    )


def fetch_latest_recommendations(account_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the latest recommendation snapshot for an account."""

    collection = _get_collection()
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        return None

    doc = collection.find_one(
        {"account_id": account_norm},
        sort=[("base_date", -1), ("updated_at", -1)],
    )
    return doc
