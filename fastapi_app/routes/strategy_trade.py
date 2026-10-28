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


@router.get("/backtest")
def get_strategy_trade_backtest(
    pool: str = Query(...),
    months: int = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """종목풀 하나의 사고팔기 백테스트 — 엔진 연결 전(UI 확인 단계)이라 명확한 오류만 낸다."""
    raise ValueError(f"사고팔기 백테스트 엔진이 아직 연결되지 않았습니다 ({pool}, 최근 {months}개월).")


@router.put("/settings")
def put_strategy_trade_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """한 종목풀의 전략 목록과 전역 슬랙 스위치를 저장하고 갱신된 화면 데이터를 반환한다.

    body: ``{"pool": "…", "strategies": [{strategy_id?·account_id·ticker + pct 3종}, …], "slack_enabled": bool}``
    strategy_id 가 없는 행은 추가, 목록에서 빠진 전략은 삭제다 (검증은 `strategy_trade_service`).
    전략 목록을 먼저 검증·저장하고, 그게 통과해야 슬랙 스위치를 바꾼다.
    """
    pool = str(payload.get("pool") or "").strip()
    if not pool or payload.get("strategies") is None or payload.get("slack_enabled") is None:
        raise ValueError("pool·strategies·slack_enabled 가 모두 필요합니다.")
    save_pool_strategies(pool, payload["strategies"])
    save_slack_enabled(bool(payload["slack_enabled"]))
    return load_strategy_trade_view(pool)


@router.post("/slack-test")
def post_strategy_trade_slack_test(_: None = Depends(require_internal_token)) -> dict:
    """지금 수동 발송(트리거 없으면 현황만). 화면 '지금 발송(테스트)' 버튼용."""
    from utils.strategy_trade_notify import notify_strategy_trade

    result = notify_strategy_trade(force=True)
    return {"sent": result["sent"], "message": result["reason"]}
