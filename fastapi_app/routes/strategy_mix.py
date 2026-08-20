"""합성 전략(SM + 신고가 50:50) 백테스트 API — `/strategy-mix` 열람 전용 화면."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token

router = APIRouter(prefix="/internal/strategy-mix", tags=["strategy-mix"])


@router.get("/meta")
def get_strategy_mix_meta(_: None = Depends(require_internal_token)) -> dict:
    """합성을 운용하는 계좌 목록 — 화면 진입 시 쓰는 가벼운 조회 (계산 없음)."""
    from utils.strategy_mix_service import mix_meta

    return mix_meta()


@router.get("/positions")
def get_strategy_mix_positions(
    pool: str | None = Query(default=None),
    as_of: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션.

    ``as_of`` 를 주면 그 날짜의 상태를 재현한다 (과거 날짜 조회).
    """
    from utils.strategy_mix_service import mix_positions

    return mix_positions(pool, as_of)


@router.get("/backtest")
def get_strategy_mix_backtest(
    pool: str | None = Query(default=None),
    months: int | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """선택한 풀의 저장 설정으로 두 전략을 백테스트해 50:50 합성 결과를 돌려준다.

    캐시는 없다 — 각 전략 화면의 백테스트와 같은 요청 시 계산이라 수 분 걸릴 수 있다.
    """
    from utils.strategy_mix_service import run_mix_backtest

    return run_mix_backtest(pool, months)


class SlackTestPayload(BaseModel):
    account_id: str


@router.post("/slack-test")
def post_slack_test(payload: SlackTestPayload, _: None = Depends(require_internal_token)) -> dict:
    """오늘의 액션을 즉시 슬랙으로 발송한다(테스트) — 변화 여부와 무관, 상태 미변경."""
    from utils.strategy_mix_notify import send_test

    try:
        return send_test(payload.account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
