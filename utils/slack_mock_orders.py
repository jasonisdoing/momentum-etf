"""주문의 슬랙 왕복 — 주문 메시지(버튼) → 확인 모달 → 웹훅 수신 → 주문·취소 → 결과 알림.

모든 액션(주문·취소)은 확인 모달을 거친다. 메시지는 상태에 따라 버튼이 바뀐다: 주문하기 → 처리 중 → 취소하기 →
결과 문구(체결·취소·거부). 끝난 메시지에는 버튼이 없지만, 하나도 체결되지 않고 취소된 주문만 다시 주문하기 버튼이 남는다.

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
from services.broker_order_service import account_env
from utils.account_settings_store import ACCOUNT_ENV_MOCK
from utils.notification import send_slack_message_v2
from utils.strategy_trade_orders import (
    FINISHED_STATUSES,
    cancel_record,
    close_warning_slack_orders,
    find_duplicate_order,
    finished_slack_orders,
    mark_slack_close_warned,
    mark_slack_notified,
    place_order,
    refresh_slack_order_accounts,
    refreshed_order,
    resolve_order,
    serialize_order,
)
from utils.strategy_trade_service import broker_account_no_for_pool

logger = logging.getLogger(__name__)

OPEN_ACTION_ID = "mock_order_open"
CANCEL_ACTION_ID = "mock_order_cancel"
ORDER_CALLBACK_ID = "mock_order_confirm"
CANCEL_CALLBACK_ID = "mock_order_cancel_confirm"
_SIDE_LABELS = {"buy": "매수", "sell": "매도"}
#: 주문 기록(`slack`)에 남기는 슬랙 위치 — 버튼이 있는 메시지(`message_ts`)와 결과를 달 스레드 뿌리(`thread_ts`).
SLACK_TARGET_KEYS = ("channel", "message_ts", "thread_ts")


def _env_word(env: str) -> str:
    """메시지에 쓰는 계좌 구분 — 운영은 실제 주문이라 눈에 띄게 적는다."""
    return "모의" if env == ACCOUNT_ENV_MOCK else "🔴 운영"


def _require_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise BrokerApiError(f"{name} 환경변수가 필요합니다.")
    return value


def _round_text(round_no: int | None) -> str:
    """신호 주문의 회차 표기 — 직접 낸 주문(회차 없음)은 비운다."""
    return f"{round_no}호 " if round_no is not None else ""


def _order_text(request: dict[str, Any]) -> str:
    side = _SIDE_LABELS[request["side"]]
    return f"{_round_text(request['round'])}{request['label']} · {request['quantity']:,}주 × 현재가 지정가 {side}"


# --- 메시지 ---------------------------------------------------------------------------------------------


def _section(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*[{_env_word(request['env'])}] {_SIDE_LABELS[request['side']]} 주문*\n{_order_text(request)}\n"
                f"버튼을 확정하는 순간의 현재가(호가 단위)로 냅니다 ({request['ticker']})"
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


def _open_blocks(request: dict[str, Any], label: str = "주문하기") -> list[dict[str, Any]]:
    side = _SIDE_LABELS[request["side"]]
    style = "primary" if request["side"] == "buy" else "danger"
    return [_section(request), _button(OPEN_ACTION_ID, f"{side} {label}", style, {"request": request})]


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
        text=f"[{_env_word(target['request']['env'])}] {_order_text(target['request'])}",
        blocks=blocks,
    )


def send_order_message(
    account_no: str, pool: str, strategy_id: str, side: str, round_no: int | None, thread_ts: str | None
) -> None:
    """주문 메시지를 슬랙으로 보낸다(thread_ts 가 있으면 그 스레드에) — 주문은 슬랙에서 버튼과 모달 확인을 거쳐야 나간다."""
    env = account_env(account_no)
    order = resolve_order(pool, strategy_id, side, round_no)
    request = {
        "env": env,
        "round": round_no,
        "account_no": account_no,
        "pool": pool,
        "strategy_id": strategy_id,
        "side": side,
        "label": order["strategy_label"],
        "ticker": order["ticker"],
        "quantity": order["quantity"],
    }
    if not send_slack_message_v2(f"[{_env_word(env)}] {_order_text(request)}", _open_blocks(request), thread_ts):
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
    guide = (
        f"주문을 확정하는 순간의 현재가(호가 단위)로 {_env_word(request['env'])} 계좌에 보냅니다. "
        "현재가 지정가라 호가에 따라 체결되지 않을 수 있습니다."
    )
    body = f"*{_order_text(request)}*\n{guide}"
    return _modal_view(ORDER_CALLBACK_ID, f"{_env_word(request['env'])} {side} 주문 확인", f"{side} 주문", body, target)


def _cancel_view(request: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    side = _SIDE_LABELS[request["side"]]
    body = f"*{_order_text(request)}*\n접수된 {_env_word(request['env'])} {side} 주문을 취소합니다(일부 체결됐다면 남은 수량만 취소됩니다)."
    return _modal_view(
        CANCEL_CALLBACK_ID, f"{_env_word(request['env'])} {side} 주문 취소 확인", "주문 취소", body, target
    )


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
    request = {**value["request"], "env": account_env(value["request"]["account_no"])}
    message = payload["message"]
    target = {
        "request": request,
        "channel": payload["channel"]["id"],
        "message_ts": message["ts"],
        # 스레드 답글이면 최상위 메시지의 ts, 최상위 메시지면 자기 ts 가 스레드의 뿌리다.
        "thread_ts": message.get("thread_ts") or message["ts"],
    }
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
    side = f"{_env_word(order['env'])} {_SIDE_LABELS[order['side']]}"
    head = f"{_round_text(order['round'])}{order['strategy_label']} {order['quantity']:,}주 × {order['price']:,}원"
    fill = f"체결 {order['filled_quantity']:,}주 · 체결가 {order['average_fill_price']:,.0f}원"
    if order["status"] == "filled":
        return f"✅ {side} 체결 완료: {head} ({fill})"
    if order["status"] == "canceled":
        return f"🚫 {side} 주문 취소됨: {head} ({fill})"
    if order["status"] == "expired":
        return f"⌛ {side} 주문 만료됨: {head} ({fill})"
    return f"❌ {side} 주문 거부됨: {head}" + (f" — {order['message']}" if order["message"] else "")


def _finish(target: dict[str, Any], order: dict[str, Any]) -> None:
    """끝난 주문의 결과를 스레드로 알리고 메시지에서 버튼을 걷어낸다 — 하나도 체결 안 된 취소만 다시 주문하기 버튼을 남긴다."""
    text = _final_text(order)
    send_slack_message_v2(text, thread_ts=target["thread_ts"])
    request = target["request"]
    if order["status"] == "canceled" and order["filled_quantity"] == 0:
        # 다시 주문해도 서버가 신호 유효성·중복·안전 장치를 처음 주문과 똑같이 검사한다.
        _update_message(target, [*_open_blocks(request, "다시 주문하기"), _note(text)])
    else:
        _update_message(target, _note_blocks(request, text))


def _place(target: dict[str, Any]) -> None:
    request = target["request"]
    side = f"{_env_word(request['env'])} {_SIDE_LABELS[request['side']]}"
    # 처리하는 동안 같은 메시지로 또 주문하지 못하게 버튼부터 걷어낸다.
    _update_message(target, _note_blocks(request, "⏳ 주문 처리 중…"))
    try:
        order = place_order(
            request["account_no"],
            request["pool"],
            request["strategy_id"],
            request["side"],
            request["round"],
            {key: target[key] for key in SLACK_TARGET_KEYS},
        )
    except Exception as exc:  # 슬랙 요청은 이미 응답했으므로 실패도 스레드로 알린다.
        logger.exception("슬랙 모의 주문 처리 실패")
        send_slack_message_v2(f"❌ {side} 주문 실패: {exc}", thread_ts=target["thread_ts"])
        _update_message(target, _open_blocks(request))
        return
    send_slack_message_v2(
        f"✅ {side} 주문 접수: {order['strategy_label']} {order['quantity']:,}주 × {order['price']:,}원 (지정가)",
        thread_ts=target["thread_ts"],
    )
    _update_message(target, _cancel_blocks(request, order["id"]))


def _cancel(target: dict[str, Any]) -> None:
    request = target["request"]
    side = f"{_env_word(request['env'])} {_SIDE_LABELS[request['side']]}"
    _update_message(target, _note_blocks(request, "⏳ 취소 처리 중…"))
    try:
        cancel_record(target["order_id"])
    except Exception as exc:
        logger.exception("슬랙 모의 주문 취소 실패")
        send_slack_message_v2(f"❌ {side} 주문 취소 실패: {exc}", thread_ts=target["thread_ts"])
        # 그 사이 주문이 끝났을 수 있다 — 끝났으면 결과 알림은 배치가 하고, 아니면 취소 버튼을 되살린다.
        order = refreshed_order(request["account_no"], target["order_id"])
        if order["status"] in FINISHED_STATUSES:
            _update_message(target, _note_blocks(request, "⏳ 결과 확인 중…"))
        else:
            _update_message(target, _cancel_blocks(request, target["order_id"]))
        return
    send_slack_message_v2(f"🚫 {side} 주문 취소를 요청했습니다.", thread_ts=target["thread_ts"])
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
        send_slack_message_v2(f"⚠️ 처리 중 오류로 멈춥니다: {exc}", thread_ts=target["thread_ts"])


def _request_of(record: dict[str, Any]) -> dict[str, Any]:
    """주문 기록으로 메시지에 그릴 주문 내용을 만든다."""
    return {
        "env": record["env"],
        "round": record["round"],
        "account_no": record["account_no"],
        "pool": record["pool"],
        "strategy_id": record["strategy_id"],
        "side": record["side"],
        "label": record["strategy_label"],
        "ticker": record["ticker"],
        "quantity": record["quantity"],
    }


def _target_of(record: dict[str, Any]) -> dict[str, Any]:
    """주문 기록으로 슬랙 메시지 위치를 만든다 — 스레드 뿌리가 없는 옛 기록은 최상위 메시지라 자기 ts 가 뿌리다."""
    slack = record["slack"]
    return {
        "request": _request_of(record),
        "channel": slack["channel"],
        "message_ts": slack["message_ts"],
        "thread_ts": slack.get("thread_ts") or slack["message_ts"],
    }


def notify_finished_orders() -> int:
    """슬랙으로 낸 주문 중 끝난 것(체결·취소·거부)을 스레드로 알리고 메시지 버튼을 걷어낸다 — 알린 건수를 돌려준다."""
    refresh_slack_order_accounts()
    records = finished_slack_orders()
    for record in records:
        _finish(_target_of(record), serialize_order(record))
        mark_slack_notified(record["_id"])
    return len(records)


def notify_close_approaching() -> int:
    """장 마감이 가까운데 체결되지 않은 슬랙 주문을 스레드로 경고하고 메시지에 취소 버튼을 다시 띄운다 — 경고한 건수.

    주문 상태는 직전의 `notify_finished_orders` 가 갱신해 둔 것을 쓴다.
    """
    records = close_warning_slack_orders()
    for record in records:
        order = serialize_order(record)
        request = _request_of(record)
        target = _target_of(record)
        side = f"{_env_word(order['env'])} {_SIDE_LABELS[order['side']]}"
        remaining = order["quantity"] - order["filled_quantity"]
        text = f"⏰ {side} 주문이 아직 체결되지 않았습니다(미체결 {remaining:,}주). 15:30 장 마감 전에 취소하지 않으면 오늘 주문은 만료됩니다."
        send_slack_message_v2(text, thread_ts=target["thread_ts"])
        _update_message(target, [*_cancel_blocks(request, order["id"]), _note(text)])
        mark_slack_close_warned(record["_id"])
    return len(records)


def send_signal_order_messages(strategy_view: dict[str, Any], triggers: list[dict[str, Any]], signal_ts: str) -> int:
    """신호가 난 회차마다 주문 메시지(버튼)를 신호 메시지의 스레드로 보낸다 — 보낸 건수.

    이미 주문이 진행 중이거나 오늘 체결된 회차는 건너뛴다(매시간 재알림에서 주문이 아직 없는 회차에만 다시 띄우려는 것).
    주문은 슬랙의 버튼과 모달 확인을 거쳐야 나간다.
    """
    account_no = broker_account_no_for_pool(strategy_view["pool"])
    pending = [
        trigger
        for trigger in triggers
        if find_duplicate_order(account_no, trigger["strategy_id"], trigger["action"], trigger["round"]) is None
    ]
    for trigger in pending:
        send_order_message(
            account_no, strategy_view["pool"], trigger["strategy_id"], trigger["action"], trigger["round"], signal_ts
        )
    return len(pending)
