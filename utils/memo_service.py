"""시스템 메모 저장소 (MongoDB `memos` 컬렉션).

제목과 함께 두 형식을 보관한다.
- `text`: 서식 없는 본문(`content`)
- `list`: 체크박스 할 일 목록(`items` = [{text, done}])

두 필드를 항상 같이 저장한다. 형식을 오갈 때 반대편 내용이 사라지지 않게 하기 위한
것이며, `type` 은 화면에 무엇을 보여줄지만 정한다.

계좌별 메모(`utils/account_notes.py`)와는 별개다 — 그쪽은 계좌마다 본문 하나뿐이고
제목이 없다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

COLLECTION = "memos"
MAX_TITLE_LENGTH = 200
MAX_ITEM_LENGTH = 500
MEMO_TYPES = ("text", "list")


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
    # `type` 이 없는 문서는 형식 기능 이전에 만들어진 것이라 텍스트형이다.
    # (마이그레이션으로 모두 채웠으므로 평상시에는 여기에 걸리지 않는다)
    memo_type = str(doc.get("type") or "text")
    items = [
        {"text": str(item.get("text") or ""), "done": bool(item.get("done"))}
        for item in (doc.get("items") or [])
        if isinstance(item, dict)
    ]
    return {
        "id": str(doc["_id"]),
        "type": memo_type if memo_type in MEMO_TYPES else "text",
        "title": str(doc.get("title") or ""),
        "content": str(doc.get("content") or ""),
        "items": items,
        "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else updated_at,
    }


def _validate(memo_type: Any, title: Any, content: Any, items: Any) -> dict[str, Any]:
    """입력을 검증해 저장할 형태로 정규화한다. 값이 이상하면 보정하지 않고 에러."""
    normalized_type = str(memo_type or "").strip()
    if normalized_type not in MEMO_TYPES:
        raise ValueError(f"'type' 은 {' 또는 '.join(MEMO_TYPES)} 이어야 합니다.")

    normalized_title = str(title or "").strip()
    if not normalized_title:
        raise ValueError("제목을 입력하세요.")
    if len(normalized_title) > MAX_TITLE_LENGTH:
        raise ValueError(f"제목은 {MAX_TITLE_LENGTH}자 이내여야 합니다.")

    if content is not None and not isinstance(content, str):
        raise ValueError("본문은 문자열이어야 합니다.")

    if items is None:
        items = []
    if not isinstance(items, list):
        raise ValueError("'items' 는 목록이어야 합니다.")
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{index + 1}번째 항목 형식이 올바르지 않습니다.")
        text = str(item.get("text") or "").strip()
        if len(text) > MAX_ITEM_LENGTH:
            raise ValueError(f"{index + 1}번째 항목은 {MAX_ITEM_LENGTH}자 이내여야 합니다.")
        normalized_items.append({"text": text, "done": bool(item.get("done"))})

    return {
        "type": normalized_type,
        "title": normalized_title,
        "content": str(content or ""),
        "items": normalized_items,
    }


def list_memos() -> list[dict[str, Any]]:
    """최근 수정 순으로 전체 메모를 반환한다."""
    return [_serialize(doc) for doc in _collection().find().sort("updated_at", -1)]


def create_memo(memo_type: Any, title: Any, content: Any, items: Any) -> dict[str, Any]:
    now = datetime.now()
    doc = {**_validate(memo_type, title, content, items), "created_at": now, "updated_at": now}
    doc["_id"] = _collection().insert_one(doc).inserted_id
    return _serialize(doc)


def update_memo(memo_id: str, memo_type: Any, title: Any, content: Any, items: Any) -> dict[str, Any]:
    updated = _collection().find_one_and_update(
        {"_id": _to_object_id(memo_id)},
        {"$set": {**_validate(memo_type, title, content, items), "updated_at": datetime.now()}},
        return_document=True,
    )
    if updated is None:
        raise LookupError(f"메모를 찾을 수 없습니다: {memo_id}")
    return _serialize(updated)


def delete_memo(memo_id: str) -> None:
    result = _collection().delete_one({"_id": _to_object_id(memo_id)})
    if result.deleted_count == 0:
        raise LookupError(f"메모를 찾을 수 없습니다: {memo_id}")
