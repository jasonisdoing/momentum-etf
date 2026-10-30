"""모의 주문의 슬랙 왕복 — 주문 메시지(버튼) → 확인 모달 → 웹훅 수신 → 주문.

주문 요청은 슬랙 버튼의 value 에 담아 보내고, 서버는 따로 저장하지 않는다. 가격·수량은 모달 표시용 사본일 뿐이고,
실제 주문은 제출 시점에 `strategy_trade_orders.place_order` 가 현재가로 다시 정한다.
슬랙은 응답을 3초 안에 받아야 하므로 모달 열기까지만 요청 안에서 하고, 주문은 응답 뒤에 처리해 결과를 스레드로 알린다.
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
from utils.strategy_trade_orders import place_order, resolve_order

logger = logging.getLogger(__name__)

OPEN_ACTION_ID = "mock_order_open"
CONFIRM_CALLBACK_ID = "mock_order_confirm"
_SIDE_LABELS = {"buy": "매수", "sell": "매도"}


def _require_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise BrokerApiError(f"{name} 환경변수가 필요합니다.")
    return value


def _order_text(request: dict[str, Any]) -> str:
    side = _SIDE_LABELS[request["side"]]
    return f"{request['label']} · {request['quantity']:,}주 × 현재가 지정가 {side}"


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
        "quantity": order["quantity"],
        "price": order["price"],
    }
    text = f"[모의투자] {_order_text(request)}"
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*[모의투자] {_SIDE_LABELS[side]} 주문*\n{_order_text(request)}\n현재가 {order['price']:,}원 ({order['ticker']})",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": OPEN_ACTION_ID,
                    "text": {"type": "plain_text", "text": f"{_SIDE_LABELS[side]} 주문하기"},
                    "style": "primary" if side == "buy" else "danger",
                    "value": json.dumps(request, ensure_ascii=False),
                }
            ],
        },
    ]
    if not send_slack_message_v2(text, blocks):
        raise BrokerApiError("슬랙으로 주문 메시지를 보내지 못했습니다.")


def _confirm_view(request: dict[str, Any], message_ts: str) -> dict[str, Any]:
    side = _SIDE_LABELS[request["side"]]
    return {
        "type": "modal",
        "callback_id": CONFIRM_CALLBACK_ID,
        "title": {"type": "plain_text", "text": f"모의 {side} 주문 확인"},
        "submit": {"type": "plain_text", "text": f"{side} 주문"},
        "close": {"type": "plain_text", "text": "취소"},
        "private_metadata": json.dumps({"request": request, "message_ts": message_ts}, ensure_ascii=False),
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{_order_text(request)}*\n"
                        f"메시지 시점 현재가 {request['price']:,}원 — 주문을 누르는 시점의 현재가(호가 단위)로 다시 정해 모의투자 계좌에 보냅니다."
                    ),
                },
            }
        ],
    }


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
    if action["action_id"] != OPEN_ACTION_ID:
        raise BrokerApiError(f"알 수 없는 슬랙 버튼입니다: {action['action_id']}")
    request = json.loads(action["value"])
    require_mock_account(request["account_no"])
    WebClient(token=_require_env("SLACK_BOT_TOKEN")).views_open(
        trigger_id=payload["trigger_id"], view=_confirm_view(request, payload["message"]["ts"])
    )


def submitted_request(payload: dict[str, Any]) -> dict[str, Any]:
    """모달 제출 payload 에서 주문 요청과 원 메시지 ts 를 꺼낸다."""
    view = payload["view"]
    if view["callback_id"] != CONFIRM_CALLBACK_ID:
        raise BrokerApiError(f"알 수 없는 슬랙 모달입니다: {view['callback_id']}")
    return json.loads(view["private_metadata"])


def place_submitted_order(submitted: dict[str, Any]) -> None:
    """확인된 주문을 보내고 결과를 원 메시지의 스레드로 알린다 — 응답 뒤 백그라운드에서 돈다."""
    request = submitted["request"]
    side = _SIDE_LABELS[request["side"]]
    try:
        order = place_order(request["account_no"], request["pool"], request["strategy_id"], request["side"])
        text = f"✅ 모의 {side} 주문 접수: {order['strategy_label']} {order['quantity']:,}주 × {order['price']:,}원 (지정가)"
    except Exception as exc:  # 슬랙 요청은 이미 응답했으므로 실패도 스레드로 알린다.
        logger.exception("슬랙 모의 주문 처리 실패")
        text = f"❌ 모의 {side} 주문 실패: {exc}"
    send_slack_message_v2(text, thread_ts=submitted["message_ts"])
