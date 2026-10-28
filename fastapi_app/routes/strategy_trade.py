"""전략 사고팔기(분할 매수/매도) 운용 현황 API — 종목풀 단위로 종목 1개씩 운용하는 전략."""

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.strategy_trade_service import (
    load_strategy_trade_view,
    resolve_etf,
    save_pool_strategies,
    save_slack_enabled,
)

router = APIRouter(prefix="/internal/strategy-trade", tags=["strategy-trade"])


@router.get("")
def get_strategy_trade(
    pool: str | None = Query(None),
    _: None = Depends(require_internal_token),
) -> dict:
    """사고팔기를 켠 종목풀 목록과, 고른 풀의 전략별 회차 현황·매수/매도 지정가를 반환한다.

    ``pool`` 을 안 주면 켜진 풀 중 첫 번째다.
    """
    return load_strategy_trade_view(pool)


@router.get("/resolve")
def get_strategy_trade_resolve(
    pool: str = Query(...),
    ticker: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """티커 확인 — 그 종목풀에 있는 ETF 면 {ticker, name}, 없으면 에러."""
    return resolve_etf(pool, ticker)


@router.put("/settings")
def put_strategy_trade_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """슬랙 스위치 또는 한 종목풀의 전략 목록을 저장하고 갱신된 화면 데이터를 반환한다.

    body 는 둘 중 하나:
      ``{"slack_enabled": true, "pool": "…"}``                     — 전역 스위치(응답에 볼 풀을 함께 준다)
      ``{"pool": "…", "strategies": [{strategy_id?·account_id·ticker + pct 3종}, …]}`` — 풀의 전략 목록 전체
    strategy_id 가 없는 행은 추가, 목록에서 빠진 전략은 삭제다 (검증은 `strategy_trade_service`).
    """
    pool = str(payload.get("pool") or "").strip() or None
    if payload.get("slack_enabled") is not None:
        save_slack_enabled(bool(payload["slack_enabled"]))
    elif payload.get("strategies") is not None:
        if pool is None:
            raise ValueError("종목풀이 필요합니다.")
        save_pool_strategies(pool, payload["strategies"])
    else:
        raise ValueError("요청에 저장할 항목이 없습니다.")
    return load_strategy_trade_view(pool)


@router.post("/slack-test")
def post_strategy_trade_slack_test(_: None = Depends(require_internal_token)) -> dict:
    """지금 수동 발송(트리거 없으면 현황만). 화면 '지금 발송(테스트)' 버튼용."""
    from utils.strategy_trade_notify import notify_strategy_trade

    result = notify_strategy_trade(force=True)
    return {"sent": result["sent"], "message": result["reason"]}
