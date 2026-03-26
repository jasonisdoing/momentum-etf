"""계좌별 메모를 MongoDB에 저장하는 유틸리티."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

_COLLECTION_NAME = "account_notes"
_HISTORY_COLLECTION_NAME = "account_note_history"
_INDEXES_ENSURED = False
_HISTORY_RETENTION_DAYS = 3


def _ensure_indexes(db) -> None:
    """메모 컬렉션 인덱스를 최초 한 번 보장한다."""
    global _INDEXES_ENSURED
    if _INDEXES_ENSURED:
        return

    note_coll = db[_COLLECTION_NAME]
    history_coll = db[_HISTORY_COLLECTION_NAME]
    try:
        note_coll.create_index([("account_id", 1)], unique=True, name="account_note_unique", background=True)
        history_coll.create_index(
            [("account_id", 1), ("saved_at", -1)],
            name="account_note_history_account_saved_at",
            background=True,
        )
        history_coll.create_index(
            [("expires_at", 1)],
            name="account_note_history_expires_at_ttl",
            expireAfterSeconds=0,
            background=True,
        )
        _INDEXES_ENSURED = True
    except Exception:
        pass


def _get_collection():
    """account_notes 컬렉션 핸들을 반환하고 최초 호출 시 인덱스를 보장한다."""
    db = get_db_connection()
    if db is None:
        return None

    _ensure_indexes(db)
    return db[_COLLECTION_NAME]


def _get_history_collection():
    """account_note_history 컬렉션 핸들을 반환한다."""
    db = get_db_connection()
    if db is None:
        return None

    _ensure_indexes(db)
    return db[_HISTORY_COLLECTION_NAME]


def _cleanup_expired_history(history_coll, *, now_utc: datetime) -> None:
    """보관 기한이 지난 메모 히스토리를 즉시 정리한다."""
    history_coll.delete_many({"expires_at": {"$lte": now_utc}})


def _archive_previous_note_if_changed(
    account_id: str,
    previous_doc: dict[str, Any] | None,
    new_content: str,
    *,
    now_utc: datetime,
) -> None:
    """이전 메모 내용이 바뀌는 경우에만 히스토리에 보관한다."""
    if not previous_doc:
        return

    previous_content = str(previous_doc.get("content") or "")
    if previous_content == new_content:
        return

    history_coll = _get_history_collection()
    if history_coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_note_history 컬렉션에 쓸 수 없습니다.")

    _cleanup_expired_history(history_coll, now_utc=now_utc)
    expires_at = now_utc + timedelta(days=_HISTORY_RETENTION_DAYS)
    history_coll.insert_one(
        {
            "account_id": account_id,
            "content": previous_content,
            "saved_at": now_utc,
            "source_updated_at": previous_doc.get("updated_at"),
            "expires_at": expires_at,
        }
    )


def load_account_note(account_id: str) -> dict[str, Any] | None:
    """계좌 메모 최신본을 반환한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_notes 컬렉션을 읽을 수 없습니다.")

    doc = coll.find_one({"account_id": account_norm}, {"_id": 0})
    if not doc:
        return None
    return {
        "account_id": account_norm,
        "content": str(doc.get("content") or ""),
        "updated_at": doc.get("updated_at"),
    }


def save_account_note(account_id: str, content: str) -> datetime:
    """계좌 메모를 저장하고 갱신 시각을 반환한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_notes 컬렉션에 쓸 수 없습니다.")

    updated_at = datetime.now(timezone.utc)
    normalized_content = str(content or "")
    previous_doc = coll.find_one({"account_id": account_norm}, {"_id": 0, "content": 1, "updated_at": 1})
    _archive_previous_note_if_changed(account_norm, previous_doc, normalized_content, now_utc=updated_at)

    payload = {
        "account_id": account_norm,
        "content": normalized_content,
        "updated_at": updated_at,
    }
    result = coll.update_one({"account_id": account_norm}, {"$set": payload}, upsert=True)
    if not result.acknowledged:
        raise RuntimeError(f"계좌 메모 저장이 확인되지 않았습니다: {account_norm}")

    logger.info("계좌 메모 저장 완료: %s", account_norm)
    return updated_at


__all__ = [
    "load_account_note",
    "save_account_note",
]
