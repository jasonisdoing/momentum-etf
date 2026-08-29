"""합성 전략(SM + 신고가 + 비워 두는 현금) 백테스트 API — `/strategy-mix` 화면."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query
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
    account_id: str | None = Query(default=None),
    as_of: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션.

    ``as_of`` 를 주면 그 날짜의 상태를 재현한다 (과거 날짜 조회).
    """
    from utils.strategy_mix_service import mix_positions

    return mix_positions(account_id, as_of)


@router.get("/backtest")
def get_strategy_mix_backtest(
    account_id: str | None = Query(default=None),
    months: int | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """선택한 계좌의 슬리브별 풀 저장 설정·배분으로 두 전략을 백테스트해 합성 결과를 돌려준다.

    캐시는 없다 — 각 전략 화면의 백테스트와 같은 요청 시 계산이라 수 분 걸릴 수 있다.
    """
    from utils.strategy_mix_service import run_mix_backtest

    return run_mix_backtest(account_id, months)


class SlackTestPayload(BaseModel):
    account_id: str


@router.post("/charts")
def post_strategy_mix_charts(
    payload: dict = Body(default={}),
    _: None = Depends(require_internal_token),
) -> dict:
    """보유 종목 일봉 + 슬리브별 기준 이평선. body: ``{"account_id": "...", "tickers": [...]}``."""
    from utils.strategy_mix_service import holding_charts_for_account

    account_id = payload.get("account_id") if isinstance(payload, dict) else None
    tickers = payload.get("tickers") if isinstance(payload, dict) else None
    if not isinstance(tickers, list):
        raise ValueError("'tickers' 는 목록이어야 합니다.")
    return {"charts": holding_charts_for_account(account_id, [str(t) for t in tickers])}


@router.post("/slack-test")
def post_slack_test(payload: SlackTestPayload, _: None = Depends(require_internal_token)) -> dict:
    """오늘의 액션을 즉시 슬랙으로 발송한다(테스트) — 변화 여부와 무관, 상태 미변경."""
    from utils.strategy_mix_notify import send_test

    try:
        return send_test(payload.account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
