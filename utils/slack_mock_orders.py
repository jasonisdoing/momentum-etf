"""모의 주문의 슬랙 왕복 — 주문 메시지(버튼) → 확인 모달 → 웹훅 수신 → 주문·취소 → 결과 알림.

모든 액션(주문·취소)은 확인 모달을 거친다. 메시지는 상태에 따라 버튼이 바뀐다: 주문하기 → 처리 중 → 취소하기 →
결과 문구(체결·취소·거부). 끝난 메시지에는 버튼이 없어 같은 메시지로 다시 주문할 수 없다.

끝난 주문의 알림은 배치(`notify_finished_orders`, 1분 간격)가 맡는다 — 서버 프로세스가 재시작돼도 끊기지 않는다.

주문 요청은 슬랙 버튼의 value 에 담아 보내고, 서버는 따로 저장하지 않는다. 가격·수량은 모달 표시용 사본일 뿐이고,
실제 주문은 제출 시점에 `strategy_trade_orders.place_order` 가 현재가로 다시 정한다.
슬랙은 응답을 3초 안에 받아야 하므로 모달 열기까지만 요청 안에서 하고, 주문·취소는 응답 뒤에 처리해 결과를 스레드로 알린다.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import parse_qs

from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier

from services.broker_api_service import BrokerApiError
from services.broker_order_service import require_mock_account
from utils.notification import send_slack_message_v2
from utils.strategy_trade_orders import (
    cancel_record,
    finished_slack_orders,
    mark_slack_notified,
    place_order,
    refresh_slack_order_accounts,
    refreshed_order,
    resolve_order,
    serialize_order,
)

logger = logging.getLogger(__name__)

OPEN_ACTION_ID = "mock_order_open"
CANCEL_ACTION_ID = "mock_order_cancel"
ORDER_CALLBACK_ID = "mock_order_confirm"
CANCEL_CALLBACK_ID = "mock_order_cancel_confirm"
_SIDE_LABELS = {"buy": "매수", "sell": "매도"}
_FINAL_STATUSES = ("filled", "canceled", "rejected")


def _require_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise BrokerApiError(f"{name} 환경변수가 필요합니다.")
    return value


def _order_text(request: dict[str, Any]) -> str:
    side = _SIDE_LABELS[request["side"]]
    return f"{request['label']} · {request['quantity']:,}주 × 현재가 지정가 {side}"


# --- 메시지 ---------------------------------------------------------------------------------------------


def _section(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*[모의투자] {_SIDE_LABELS[request['side']]} 주문*\n{_order_text(request)}\n"
                f"현재가 {request['price']:,}원 ({request['ticker']})"
            ),
        },
    }


def _button(action_id: str, text: str, style: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "action_id": action_id,
                "text": {"type": "plain_text", "text": text},
                "style": style,
                "value": json.dumps(value, ensure_ascii=False),
            }
        ],
    }


def _note(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def _open_blocks(request: dict[str, Any]) -> list[dict[str, Any]]:
    side = _SIDE_LABELS[request["side"]]
    style = "primary" if request["side"] == "buy" else "danger"
    return [_section(request), _button(OPEN_ACTION_ID, f"{side} 주문하기", style, {"request": request})]


def _cancel_blocks(request: dict[str, Any], order_id: str) -> list[dict[str, Any]]:
    side = _SIDE_LABELS[request["side"]]
    return [
        _section(request),
        _button(CANCEL_ACTION_ID, f"{side} 주문 취소하기", "danger", {"request": request, "order_id": order_id}),
    ]


def _note_blocks(request: dict[str, Any], note: str) -> list[dict[str, Any]]:
    return [_section(request), _note(note)]


def _update_message(target: dict[str, Any], blocks: list[dict[str, Any]]) -> None:
    """원 메시지의 버튼·상태 줄을 교체한다."""
    WebClient(token=_require_env("SLACK_BOT_TOKEN")).chat_update(
        channel=target["channel"],
        ts=target["message_ts"],
        text=f"[모의투자] {_order_text(target['request'])}",
        blocks=blocks,
    )


def send_order_message(account_no: str, pool: str, strategy_id: str, side: str) -> None:
    """모의 주문 메시지를 슬랙으로 보낸다 — 주문은 슬랙에서 버튼과 모달 확인을 거쳐야 나간다."""
    require_mock_account(account_no)
    order = resolve_order(pool, strategy_id, side)
    request = {
        "account_no": account_no,
        "pool": pool,
        "strategy_id": strategy_id,
        "side": side,
        "label": order["strategy_label"],
        "ticker": order["ticker"],
        "quantity": order["quantity"],
        "price": order["price"],
    }
    if not send_slack_message_v2(f"[모의투자] {_order_text(request)}", _open_blocks(request)):
        raise BrokerApiError("슬랙으로 주문 메시지를 보내지 못했습니다.")


# --- 모달 -----------------------------------------------------------------------------------------------


def _modal_view(callback_id: str, title: str, submit: str, body: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "modal",
        "callback_id": callback_id,
        "title": {"type": "plain_text", "text": title},
        "submit": {"type": "plain_text", "text": submit},
        "close": {"type": "plain_text", "text": "닫기"},
        "private_metadata": json.dumps(metadata, ensure_ascii=False),
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": body}}],
    }


def _order_view(request: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    side = _SIDE_LABELS[request["side"]]
    body = (
        f"*{_order_text(request)}*\n"
        f"메시지 시점 현재가 {request['price']:,}원 — 주문을 누르는 시점의 현재가(호가 단위)로 다시 정해 모의투자 계좌에 보냅니다."
    )
    return _modal_view(ORDER_CALLBACK_ID, f"모의 {side} 주문 확인", f"{side} 주문", body, target)


def _cancel_view(request: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    side = _SIDE_LABELS[request["side"]]
    body = f"*{_order_text(request)}*\n접수된 모의 {side} 주문을 취소합니다(일부 체결됐다면 남은 수량만 취소됩니다)."
    return _modal_view(CANCEL_CALLBACK_ID, f"모의 {side} 주문 취소 확인", "주문 취소", body, target)


def verify_signature(raw_body: str, timestamp: str, signature: str) -> None:
    """슬랙이 보낸 요청인지 서명으로 확인한다 — 시각이 5분 넘게 어긋난 요청도 거절한다."""
    verifier = SignatureVerifier(_require_env("SLACK_SIGNING_SECRET"))
    if not verifier.is_valid(body=raw_body, timestamp=timestamp, signature=signature):
        raise BrokerApiError("슬랙 서명 검증에 실패했습니다.")


def parse_payload(raw_body: str) -> dict[str, Any]:
    values = parse_qs(raw_body).get("payload")
    if not values:
        raise BrokerApiError("슬랙 요청에 payload 가 없습니다.")
    return json.loads(values[0])


def open_confirm_modal(payload: dict[str, Any]) -> None:
    """버튼을 누른 사용자에게 확인 모달을 연다 — trigger_id 는 3초 안에 써야 한다."""
    action = payload["actions"][0]
    value = json.loads(action["value"])
    request = value["request"]
    require_mock_account(request["account_no"])
    target = {"request": request, "channel": payload["channel"]["id"], "message_ts": payload["message"]["ts"]}
    if action["action_id"] == OPEN_ACTION_ID:
        view = _order_view(request, target)
    elif action["action_id"] == CANCEL_ACTION_ID:
        view = _cancel_view(request, {**target, "order_id": value["order_id"]})
    else:
        raise BrokerApiError(f"알 수 없는 슬랙 버튼입니다: {action['action_id']}")
    WebClient(token=_require_env("SLACK_BOT_TOKEN")).views_open(trigger_id=payload["trigger_id"], view=view)


def submitted_request(payload: dict[str, Any]) -> dict[str, Any]:
    """모달 제출 payload 에서 처리할 일(주문·취소)과 대상 메시지를 꺼낸다."""
    view = payload["view"]
    kinds = {ORDER_CALLBACK_ID: "order", CANCEL_CALLBACK_ID: "cancel"}
    if view["callback_id"] not in kinds:
        raise BrokerApiError(f"알 수 없는 슬랙 모달입니다: {view['callback_id']}")
    return {**json.loads(view["private_metadata"]), "kind": kinds[view["callback_id"]]}


# --- 처리 -----------------------------------------------------------------------------------------------


def _final_text(order: dict[str, Any]) -> str:
    side = _SIDE_LABELS[order["side"]]
    head = f"{order['strategy_label']} {side} {order['quantity']:,}주 × {order['price']:,}원"
    if order["status"] == "filled":
        return f"✅ 모의 {side} 체결 완료: {head}"
    if order["status"] == "canceled":
        return f"🚫 모의 {side} 주문 취소됨: {head} (체결 {order['filled_quantity']:,}주)"
    return f"❌ 모의 {side} 주문 거부됨: {head}" + (f" — {order['message']}" if order["message"] else "")


def _finish(target: dict[str, Any], order: dict[str, Any]) -> None:
    """끝난 주문의 결과를 스레드로 알리고 메시지에서 버튼을 걷어낸다."""
    text = _final_text(order)
    send_slack_message_v2(text, thread_ts=target["message_ts"])
    _update_message(target, _note_blocks(target["request"], text))


def _place(target: dict[str, Any]) -> None:
    request = target["request"]
    side = _SIDE_LABELS[request["side"]]
    # 처리하는 동안 같은 메시지로 또 주문하지 못하게 버튼부터 걷어낸다.
    _update_message(target, _note_blocks(request, "⏳ 주문 처리 중…"))
    try:
        order = place_order(
            request["account_no"],
            request["pool"],
            request["strategy_id"],
            request["side"],
            {"channel": target["channel"], "message_ts": target["message_ts"]},
        )
    except Exception as exc:  # 슬랙 요청은 이미 응답했으므로 실패도 스레드로 알린다.
        logger.exception("슬랙 모의 주문 처리 실패")
        send_slack_message_v2(f"❌ 모의 {side} 주문 실패: {exc}", thread_ts=target["message_ts"])
        _update_message(target, _open_blocks(request))
        return
    send_slack_message_v2(
        f"✅ 모의 {side} 주문 접수: {order['strategy_label']} {order['quantity']:,}주 × {order['price']:,}원 (지정가)",
        thread_ts=target["message_ts"],
    )
    _update_message(target, _cancel_blocks(request, order["id"]))


def _cancel(target: dict[str, Any]) -> None:
    request = target["request"]
    side = _SIDE_LABELS[request["side"]]
    _update_message(target, _note_blocks(request, "⏳ 취소 처리 중…"))
    try:
        cancel_record(target["order_id"])
    except Exception as exc:
        logger.exception("슬랙 모의 주문 취소 실패")
        send_slack_message_v2(f"❌ 모의 {side} 주문 취소 실패: {exc}", thread_ts=target["message_ts"])
        # 그 사이 주문이 끝났을 수 있다 — 끝났으면 결과 알림은 배치가 하고, 아니면 취소 버튼을 되살린다.
        order = refreshed_order(request["account_no"], target["order_id"])
        if order["status"] in _FINAL_STATUSES:
            _update_message(target, _note_blocks(request, "⏳ 결과 확인 중…"))
        else:
            _update_message(target, _cancel_blocks(request, target["order_id"]))
        return
    send_slack_message_v2(f"🚫 모의 {side} 주문 취소를 요청했습니다.", thread_ts=target["message_ts"])
    # 취소 결과(체결 수량 포함)로 메시지를 마무리하는 일은 배치(`notify_finished_orders`)가 한다.


def run_submission(target: dict[str, Any]) -> None:
    """확인된 주문·취소를 처리한다 — 응답 뒤 백그라운드에서 돈다."""
    try:
        if target["kind"] == "order":
            _place(target)
        else:
            _cancel(target)
    except Exception as exc:
        logger.exception("슬랙 모의 주문 후속 처리 실패")
        send_slack_message_v2(f"⚠️ 처리 중 오류로 멈춥니다: {exc}", thread_ts=target["message_ts"])


def _request_of(record: dict[str, Any]) -> dict[str, Any]:
    """주문 기록으로 메시지에 그릴 주문 내용을 만든다."""
    return {
        "account_no": record["account_no"],
        "pool": record["pool"],
        "strategy_id": record["strategy_id"],
        "side": record["side"],
        "label": record["strategy_label"],
        "ticker": record["ticker"],
        "quantity": record["quantity"],
        "price": record["price"],
    }


def notify_finished_orders() -> int:
    """슬랙으로 낸 주문 중 끝난 것(체결·취소·거부)을 스레드로 알리고 메시지 버튼을 걷어낸다 — 알린 건수를 돌려준다."""
    refresh_slack_order_accounts()
    records = finished_slack_orders()
    for record in records:
        target = {
            "request": _request_of(record),
            "channel": record["slack"]["channel"],
            "message_ts": record["slack"]["message_ts"],
        }
        _finish(target, serialize_order(record))
        mark_slack_notified(record["_id"])
    return len(records)
