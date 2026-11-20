"""합성 전략 운용 API — `/strategy-mix` 화면(계좌 목록·운용 현황·차트·슬랙 테스트)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, PositiveInt

from config import HOLDING_CHART_MONTHS
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
    _: None = Depends(require_internal_token),
) -> dict:
    """오늘 기준 합성 운영 상태 — 보유 목록(목표 비중)·현금 비중·오늘의 액션."""
    from utils.strategy_mix_orders import can_order_account, mix_order_signals
    from utils.strategy_mix_positions import mix_positions

    positions = mix_positions(account_id)
    can_order = can_order_account(positions["account_id"])
    positions["can_order"] = can_order
    positions["order_signals"] = (
        mix_order_signals(positions["account_id"], positions["actions"]["groups"]) if can_order else {}
    )
    return positions


class SlackTestPayload(BaseModel):
    account_id: str


@router.post("/charts")
def post_strategy_mix_charts(
    payload: dict = Body(default={}),
    _: None = Depends(require_internal_token),
) -> dict:
    """보유 종목 일봉 + 슬리브별 기준 이평선. body: ``{"account_id": "...", "tickers": [...]}``."""
    from utils.strategy_mix_positions import holding_charts_for_account

    account_id = payload.get("account_id") if isinstance(payload, dict) else None
    tickers = payload.get("tickers") if isinstance(payload, dict) else None
    if not isinstance(tickers, list):
        raise ValueError("'tickers' 는 목록이어야 합니다.")
    return {
        "charts": holding_charts_for_account(account_id, [str(t) for t in tickers]),
        # 화면 안내 문구("최근 N개월 일봉입니다")가 쓰는 값 — 프론트에 복사본을 두지 않는다.
        "months": HOLDING_CHART_MONTHS,
    }


@router.post("/slack-test")
def post_slack_test(payload: SlackTestPayload, _: None = Depends(require_internal_token)) -> dict:
    """오늘의 액션을 즉시 슬랙으로 발송한다(테스트) — 변화 여부와 무관, 상태 미변경."""
    from utils.strategy_mix_notify import send_test

    try:
        return send_test(payload.account_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class OrderSignalPayload(BaseModel):
    account_id: str
    action_key: str
    side: Literal["buy", "sell"]
    quantity: PositiveInt


@router.post("/order-signal")
def post_order_signal(payload: OrderSignalPayload, _: None = Depends(require_internal_token)) -> dict:
    """합성 액션(오늘·내일) 한 건을 Slack 주문 신호로 보낸다."""
    from services.broker_api_service import BrokerApiError
    from utils.strategy_mix_orders import send_mix_order_signal

    try:
        return send_mix_order_signal(**payload.model_dump())
    except (BrokerApiError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class WithdrawOrderSignalPayload(BaseModel):
    account_id: str
    ticker: str
    key: str


@router.post("/withdraw-signal")
def post_withdraw_signal(payload: WithdrawOrderSignalPayload, _: None = Depends(require_internal_token)) -> dict:
    """합성의 미사용 주문 신호 한 건을 회수한다."""
    from services.broker_api_service import BrokerApiError
    from utils.strategy_mix_orders import withdraw_mix_order_signal

    try:
        withdraw_mix_order_signal(payload.account_id, payload.ticker, payload.key)
    except (BrokerApiError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get("/signal-action")
def get_signal_action(
    account_id: str = Query(...),
    ticker: str = Query(...),
    action_key: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """웹 통합 모달에서 슬랙 발송 신호의 실시간 상태·미리보기를 조회한다."""
    from services.broker_api_service import BrokerApiError
    from utils.strategy_mix_orders import get_mix_signal_action_state

    try:
        return get_mix_signal_action_state(account_id, ticker, action_key)
    except (BrokerApiError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class MixSignalActionPayload(BaseModel):
    kind: Literal["order", "modify", "cancel"]
    account_id: str
    ticker: str
    action_key: str
    thread_ts: str | None = None
    confirmation_id: str | None = None
    confirmed: dict | None = None
    order_id: str | None = None


@router.post("/signal-action")
def post_signal_action(payload: MixSignalActionPayload, _: None = Depends(require_internal_token)) -> dict:
    """웹 통합 모달에서 슬랙 주문·호가정정·취소를 실행한다."""
    from services.broker_api_service import BrokerApiError
    from utils.strategy_mix_orders import (
        execute_mix_web_cancel,
        execute_mix_web_modify,
        execute_mix_web_order,
    )

    try:
        if payload.kind == "order":
            if not payload.thread_ts or not payload.confirmation_id or not payload.confirmed:
                raise ValueError("주문 실행에 필요한 확인 정보가 부족합니다.")
            return execute_mix_web_order(
                payload.account_id,
                payload.ticker,
                payload.action_key,
                payload.thread_ts,
                payload.confirmation_id,
                payload.confirmed,
            )
        if not payload.order_id:
            raise ValueError("대상 주문 ID가 없습니다.")
        if payload.kind == "modify":
            return execute_mix_web_modify(payload.account_id, payload.ticker, payload.action_key, payload.order_id)
        return execute_mix_web_cancel(payload.account_id, payload.ticker, payload.action_key, payload.order_id)
    except (BrokerApiError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
