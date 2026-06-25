from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.system_service import (
    BatchAlreadyRunningError,
    JobCancelForbiddenError,
    SystemAction,
    extract_job_logs_for_run,
    load_system_data,
    request_cancel_running_job,
    trigger_system_action,
)

router = APIRouter(prefix="/internal/system", tags=["system"])


class SystemActionRequest(BaseModel):
    action: SystemAction


@router.get("")
def get_system_data(_: None = Depends(require_internal_token)) -> dict[str, object]:
    return load_system_data()


@router.post("")
def post_system_action(payload: SystemActionRequest, _: None = Depends(require_internal_token)) -> dict[str, str]:
    try:
        return {"message": trigger_system_action(payload.action)}
    except BatchAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/cancel")
def post_cancel_job(
    payload: dict[str, str],
    _: None = Depends(require_internal_token),
) -> dict[str, str]:
    """현재 fastapi 인스턴스의 APP_TYPE 과 일치하는 worker 가 처리 중일 때만 취소 가능."""
    job_key = str(payload.get("key") or "").strip()
    if not job_key:
        raise HTTPException(status_code=400, detail="key 필드가 필요합니다.")
    try:
        return request_cancel_running_job(job_key)
    except JobCancelForbiddenError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/job-logs")
def get_job_logs(
    key: str = Query(..., description="배치 작업 키 (예: cache_refresh)"),
    started_at: str = Query(..., description="실행 시작 ISO 시각"),
    ended_at: str | None = Query(None, description="실행 종료 ISO 시각 (없으면 시작+30분)"),
    _: None = Depends(require_internal_token),
) -> PlainTextResponse:
    """logs/YYYY-MM-DD.log 에서 해당 작업 실행 구간의 라인을 추출해 다운로드용으로 반환.

    같은 fastapi 인스턴스가 호스팅하는 호스트의 logs 만 접근 가능. 다른 인스턴스
    (서버 ↔ 로컬) 에서 실행된 작업은 그 인스턴스의 페이지에서 다운로드해야 한다.
    """
    content = extract_job_logs_for_run(key, started_at, ended_at)
    if content is None:
        raise HTTPException(status_code=404, detail="해당 일자의 로그 파일이 없습니다.")
    safe_started = started_at.replace(":", "").replace("+", "_")
    filename = f"{key}_{safe_started}.log"
    return PlainTextResponse(
        content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
