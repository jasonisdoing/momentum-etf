"""계좌별 투두 아이템을 MongoDB에 저장하는 유틸리티."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bson import ObjectId

from utils.db_manager import get_db_connection

_COLLECTION_NAME = "account_todos"
_INDEX_ENSURED = False


def _get_collection():
    """account_todos 컬렉션 핸들을 반환하고 최초 호출 시 인덱스를 보장한다."""
    global _INDEX_ENSURED
    db = get_db_connection()
    if db is None:
        return None

    coll = db[_COLLECTION_NAME]
    if not _INDEX_ENSURED:
        try:
            coll.create_index([("account_id", 1), ("status", 1), ("created_at", -1)], background=True)
            _INDEX_ENSURED = True
        except Exception:
            pass
    return coll


def list_account_todos(account_id: str) -> list[dict[str, Any]]:
    """계좌별 투두 목록을 정렬된 상태로 반환한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_todos 컬렉션을 읽을 수 없습니다.")

    docs = list(coll.find({"account_id": account_norm}, {"account_id": 0}))

    results: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = doc.pop("_id", None)
        results.append(
            {
                "todo_id": str(doc_id) if doc_id is not None else "",
                "content": str(doc.get("content") or ""),
                "status": str(doc.get("status") or "open"),
                "created_at": doc.get("created_at"),
                "completed_at": doc.get("completed_at"),
                "updated_at": doc.get("updated_at"),
            }
        )

    def _sort_key(item: dict[str, Any]) -> tuple[int, float]:
        is_done = str(item.get("status") or "") == "done"
        created_at = item.get("created_at")
        if isinstance(created_at, datetime):
            created_ts = created_at.timestamp()
        else:
            created_ts = 0.0
        return (1 if is_done else 0, -created_ts)

    return sorted(results, key=_sort_key)


def create_account_todo(account_id: str, content: str = "") -> str:
    """새 투두 아이템을 생성하고 ID를 반환한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_todos 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    result = coll.insert_one(
        {
            "account_id": account_norm,
            "content": str(content or ""),
            "status": "open",
            "created_at": now,
            "completed_at": None,
            "updated_at": now,
        }
    )
    return str(result.inserted_id)


def update_account_todo_content(account_id: str, todo_id: str, content: str) -> datetime:
    """투두 내용을 저장하고 갱신 시각을 반환한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_todos 컬렉션에 쓸 수 없습니다.")

    updated_at = datetime.now(timezone.utc)
    result = coll.update_one(
        {"_id": ObjectId(todo_id), "account_id": account_norm},
        {"$set": {"content": str(content or ""), "updated_at": updated_at}},
    )
    if not result.matched_count:
        raise RuntimeError("저장할 투두 아이템을 찾지 못했습니다.")
    return updated_at


def complete_account_todo(account_id: str, todo_id: str) -> datetime:
    """투두를 완료 상태로 변경한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_todos 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    result = coll.update_one(
        {"_id": ObjectId(todo_id), "account_id": account_norm},
        {"$set": {"status": "done", "completed_at": now, "updated_at": now}},
    )
    if not result.matched_count:
        raise RuntimeError("완료 처리할 투두 아이템을 찾지 못했습니다.")
    return now


def reopen_account_todo(account_id: str, todo_id: str) -> datetime:
    """완료된 투두를 다시 진행 중 상태로 변경한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_todos 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    result = coll.update_one(
        {"_id": ObjectId(todo_id), "account_id": account_norm},
        {"$set": {"status": "open", "completed_at": None, "updated_at": now}},
    )
    if not result.matched_count:
        raise RuntimeError("완료취소할 투두 아이템을 찾지 못했습니다.")
    return now


def delete_account_todo(account_id: str, todo_id: str) -> None:
    """투두를 완전히 삭제한다."""
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id가 필요합니다.")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — account_todos 컬렉션에 쓸 수 없습니다.")

    result = coll.delete_one({"_id": ObjectId(todo_id), "account_id": account_norm})
    if not result.deleted_count:
        raise RuntimeError("삭제할 투두 아이템을 찾지 못했습니다.")


__all__ = [
    "list_account_todos",
    "create_account_todo",
    "update_account_todo_content",
    "complete_account_todo",
    "reopen_account_todo",
    "delete_account_todo",
]
