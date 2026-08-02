"""시스템 메모 저장소 (MongoDB `memos` 컬렉션).

제목 + 서식 없는 본문을 여러 개 보관한다. 계좌별 메모(`utils/account_notes.py`)와는
별개다 — 그쪽은 계좌마다 본문 하나뿐이고 제목이 없다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

COLLECTION = "memos"
MAX_TITLE_LENGTH = 200


def _collection():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 메모를 처리할 수 없습니다.")
    return db[COLLECTION]


def _to_object_id(memo_id: str) -> ObjectId:
    try:
        return ObjectId(str(memo_id))
    except (InvalidId, TypeError) as error:
        raise ValueError(f"잘못된 메모 id 입니다: {memo_id}") from error


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    updated_at = doc.get("updated_at")
    return {
        "id": str(doc["_id"]),
        "title": str(doc.get("title") or ""),
        "content": str(doc.get("content") or ""),
        "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else updated_at,
    }


def _validate(title: Any, content: Any) -> tuple[str, str]:
    """제목·본문을 검증해 정규화한다. 비어 있으면 임의 값으로 채우지 않고 에러."""
    normalized_title = str(title or "").strip()
    if not normalized_title:
        raise ValueError("제목을 입력하세요.")
    if len(normalized_title) > MAX_TITLE_LENGTH:
        raise ValueError(f"제목은 {MAX_TITLE_LENGTH}자 이내여야 합니다.")
    if content is not None and not isinstance(content, str):
        raise ValueError("본문은 문자열이어야 합니다.")
    return normalized_title, str(content or "")


def list_memos() -> list[dict[str, Any]]:
    """최근 수정 순으로 전체 메모를 반환한다."""
    return [_serialize(doc) for doc in _collection().find().sort("updated_at", -1)]


def create_memo(title: Any, content: Any) -> dict[str, Any]:
    normalized_title, normalized_content = _validate(title, content)
    now = datetime.now()
    doc = {
        "title": normalized_title,
        "content": normalized_content,
        "created_at": now,
        "updated_at": now,
    }
    doc["_id"] = _collection().insert_one(doc).inserted_id
    return _serialize(doc)


def update_memo(memo_id: str, title: Any, content: Any) -> dict[str, Any]:
    normalized_title, normalized_content = _validate(title, content)
    updated = _collection().find_one_and_update(
        {"_id": _to_object_id(memo_id)},
        {
            "$set": {
                "title": normalized_title,
                "content": normalized_content,
                "updated_at": datetime.now(),
            }
        },
        return_document=True,
    )
    if updated is None:
        raise LookupError(f"메모를 찾을 수 없습니다: {memo_id}")
    return _serialize(updated)


def delete_memo(memo_id: str) -> None:
    result = _collection().delete_one({"_id": _to_object_id(memo_id)})
    if result.deleted_count == 0:
        raise LookupError(f"메모를 찾을 수 없습니다: {memo_id}")
