"""시스템 메모 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.memo_service import create_memo, delete_memo, list_memos, update_memo

router = APIRouter(prefix="/internal/memos", tags=["memos"])


def _payload_fields(payload: dict) -> tuple[object, object]:
    if not isinstance(payload, dict):
        raise ValueError("요청 본문이 올바르지 않습니다.")
    return payload.get("title"), payload.get("content")


@router.get("")
def get_memos(_: None = Depends(require_internal_token)) -> dict:
    """전체 메모를 최근 수정 순으로 반환한다."""
    return {"memos": list_memos()}


@router.post("")
def post_memo(payload: dict = Body(...), _: None = Depends(require_internal_token)) -> dict:
    """새 메모를 만든다. body: ``{"title": "...", "content": "..."}``."""
    title, content = _payload_fields(payload)
    return {"memo": create_memo(title, content)}


@router.put("/{memo_id}")
def put_memo(
    memo_id: str, payload: dict = Body(...), _: None = Depends(require_internal_token)
) -> dict:
    """기존 메모의 제목·본문을 교체한다."""
    title, content = _payload_fields(payload)
    return {"memo": update_memo(memo_id, title, content)}


@router.delete("/{memo_id}")
def remove_memo(memo_id: str, _: None = Depends(require_internal_token)) -> dict:
    """메모를 삭제한다 (휴지통 없이 즉시 하드 딜리트)."""
    delete_memo(memo_id)
    return {"deleted": memo_id}
