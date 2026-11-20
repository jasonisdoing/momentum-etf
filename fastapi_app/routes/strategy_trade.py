"""전략 사고팔기(분할 매수/매도) 운용 현황 API — 계좌별 종목 전략."""

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from fastapi_app.dependencies import require_internal_token
from services.broker_api_service import BrokerApiError
from utils.strategy_trade_notify import mark_sent_rounds, preview_manual_signal, send_manual_signal, send_test
from utils.strategy_trade_orders import order_limit_usage
from utils.strategy_trade_service import (
    load_strategy_trade_view,
    save_account_trade,
)
from utils.strategy_trade_threads import withdraw_signal_by_key

router = APIRouter(prefix="/internal/strategy-trade", tags=["strategy-trade"])


def _screen_view(account_id: str | None) -> dict:
    """화면 데이터 — 운용 현황에 슬랙 신호 표시와 오늘 API 주문 한도·사용량(`order_limit`)을 붙인다."""
    view = mark_sent_rounds(load_strategy_trade_view(account_id))
    view["order_limit"] = order_limit_usage(view["account_id"]) if view["account_id"] else None
    return view


@router.get("")
def get_strategy_trade(
    account_id: str | None = Query(None),
    _: None = Depends(require_internal_token),
) -> dict:
    """사고팔기 계좌 목록과, 고른 계좌의 전략별 회차 현황·매수/매도 지정가를 반환한다.

    ``account_id`` 를 안 주면 첫 계좌다.
    """
    return _screen_view(account_id)


@router.put("/settings")
def put_strategy_trade_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """한 계좌의 슬랙 설정과 전략 목록을 저장하고 갱신된 화면 데이터를 반환한다."""
    account_id = str(payload.get("account_id") or "").strip()
    required = ("slack_enabled", "recommended_gap_mode")
    if not account_id or payload.get("strategies") is None or any(key not in payload for key in required):
        raise ValueError("account_id·slack_enabled·recommended_gap_mode·strategies 가 모두 필요합니다.")
    save_account_trade(
        account_id,
        payload["slack_enabled"],
        payload["recommended_gap_mode"],
        payload["strategies"],
        payload.get("groups"),
    )
    return _screen_view(account_id)


@router.post("/withdraw-signal")
def post_strategy_trade_withdraw_signal(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """슬랙으로 나간 신호 한 건을 회수한다 — 주문이 없을 때만.

    body: ``{"key": 화면이 준 발송 기록 키}``. 본문은 표준 회수 한 줄로 바뀌고 앱 답글과 발송 기록은 지운다.
    """
    key = str(payload.get("key") or "").strip()
    if not key:
        raise ValueError("key 가 필요합니다.")
    return withdraw_signal_by_key(key)


@router.post("/slack-test")
def post_strategy_trade_slack_test(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """그 계좌의 사고팔기 전략 현재 상태를 슬랙으로 바로 보낸다(스위치와 무관한 테스트).

    body: ``{"account_id": "…"}``.
    """
    account_id = str(payload.get("account_id") or "").strip()
    if not account_id:
        raise ValueError("account_id 가 필요합니다.")
    result = send_test(account_id)
    if not result["sent"]:
        raise ValueError(result["reason"])
    return {"sent": True, "count": len(result["triggers"])}


@router.post("/manual-signal/preview")
def post_manual_signal_preview(payload: dict = Body(...), _: None = Depends(require_internal_token)) -> dict:
    """회차 신호의 계좌·종목·가격·수량을 서버에서 확인한다."""
    try:
        return preview_manual_signal(payload)
    except BrokerApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/manual-signal")
def post_manual_signal(payload: dict = Body(...), _: None = Depends(require_internal_token)) -> dict:
    """확인한 수동 신호를 Slack으로만 보낸다."""
    try:
        return send_manual_signal(payload)
    except BrokerApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_trade_thread_ts(key_or_ts: str) -> str:
    from utils.strategy_trade_notify import _load_sent_log

    entry = _load_sent_log().get(key_or_ts)
    if entry and entry.get("signal_ts"):
        return str(entry["signal_ts"])
    return key_or_ts


@router.get("/signal-action")
def get_trade_signal_action(
    account_id: str,
    thread_ts: str,
    side: str | None = None,
    round_no: int | None = None,
    ticker: str | None = None,
    _: None = Depends(require_internal_token),
) -> dict:
    """전략 사고팔기 웹 통합 모달에서 슬랙 신호의 실시간 상태를 조회한다."""
    from utils.slack_mock_orders import describe_slack_thread_action

    resolved_ts = _resolve_trade_thread_ts(thread_ts)
    try:
        state, _, _ = describe_slack_thread_action(
            account_id,
            resolved_ts,
            side=side,
            round_no=round_no,
            ticker=ticker,
            allow_unregistered_withdraw=False,
        )
        return state
    except (BrokerApiError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/signal-action")
def post_trade_signal_action(payload: dict = Body(...), _: None = Depends(require_internal_token)) -> dict:
    """전략 사고팔기 웹 통합 모달에서 슬랙 주문·호가정정·취소를 실행한다."""
    from utils.slack_mock_orders import describe_slack_thread_action, execute_slack_thread_action

    kind = str(payload.get("kind") or "").strip()
    account_id = str(payload.get("account_id") or "").strip()
    raw_ts = str(payload.get("thread_ts") or "").strip()
    side = payload.get("side")
    round_no = payload.get("round_no")
    ticker = payload.get("ticker")
    order_id = payload.get("order_id")
    if not kind or not account_id or not raw_ts:
        raise HTTPException(status_code=400, detail="kind·account_id·thread_ts 가 필요합니다.")
    resolved_ts = _resolve_trade_thread_ts(raw_ts)
    try:
        execute_slack_thread_action(
            kind,
            account_id,
            resolved_ts,
            order_id=str(order_id) if order_id else None,
            ticker=str(ticker) if ticker else None,
            side=str(side) if side else None,
            round_no=int(round_no) if round_no is not None else None,
        )
        state, _, _ = describe_slack_thread_action(
            account_id,
            resolved_ts,
            side=str(side) if side else None,
            round_no=int(round_no) if round_no is not None else None,
            ticker=str(ticker) if ticker else None,
            allow_unregistered_withdraw=False,
        )
        return state
    except (BrokerApiError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
