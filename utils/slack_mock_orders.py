"""주문의 슬랙 왕복 — 주문 메시지(버튼) → 확인 모달 → 웹훅 수신 → 주문·취소 → 결과 알림.

모든 액션(주문·취소)은 확인 모달을 거친다. 메시지는 상태에 따라 버튼이 바뀐다: 주문하기 → 처리 중 → 취소하기 →
결과 문구(체결·취소·거부). 끝난 메시지에는 버튼이 없지만, 하나도 체결되지 않고 취소된 주문만 다시 주문하기 버튼이 남는다.

끝난 주문의 알림은 배치(`notify_finished_orders`, 1분 간격)가 맡는다 — 서버 프로세스가 재시작돼도 끊기지 않는다.

주문 요청은 슬랙 버튼의 value 에 담아 보내고, 서버는 따로 저장하지 않는다. 가격·수량은 모달 표시용 사본일 뿐이고,
실제 주문은 제출 시점에 `strategy_trade_orders.place_order` 가 상대 1호가(매수는 매도 1호가, 매도는 매수 1호가)로 다시 정한다.
슬랙은 응답을 3초 안에 받아야 하므로 모달 열기까지만 요청 안에서 하고, 주문·취소는 응답 뒤에 처리해 결과를 스레드로 알린다.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import parse_qs

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

from services.broker_api_service import BrokerApiError
from services.broker_trade_service import account_env
from utils.account_settings_store import ACCOUNT_ENV_MOCK
from utils.settings_loader import get_slack_channel
from utils.strategy_trade_orders import (
    FINISHED_STATUSES,
    RoundNotOrderableError,
    cancel_record,
    close_warning_slack_orders,
    filled_signal_threads,
    find_duplicate_order,
    finished_slack_orders,
    mark_signal_thread_completed,
    mark_slack_close_warned,
    mark_slack_notified,
    pair_records,
    place_order,
    place_swap,
    refresh_slack_order_accounts,
    refreshed_order,
    register_signal_thread,
    resolve_order,
    resolve_swap,
    serialize_order,
    venue_close_time,
)
from utils.strategy_trade_threads import send_thread_message

logger = logging.getLogger(__name__)

OPEN_ACTION_ID = "mock_order_open"
CANCEL_ACTION_ID = "mock_order_cancel"
ORDER_CALLBACK_ID = "mock_order_confirm"
CANCEL_CALLBACK_ID = "mock_order_cancel_confirm"
_SIDE_LABELS = {"buy": "매수", "sell": "매도"}
#: 주문 가격으로 쓰는 상대 1호가 — 매수는 파는 쪽(매도 1호가)에, 매도는 사는 쪽(매수 1호가)에 맞춘다.
_QUOTE_LABELS = {"buy": "매도 1호가", "sell": "매수 1호가"}
#: 주문 기록(`slack`)에 남기는 슬랙 위치 — 버튼이 있는 메시지(`message_ts`)와 결과를 달 스레드 뿌리(`thread_ts`).
SLACK_TARGET_KEYS = ("channel", "message_ts", "thread_ts")
# 신호 메시지 반응 — 스레드에서 첫 주문이 접수되면 모래시계, 주문 메시지가 전부 체결되면 완료 표시,
# 주문하려는 순간 그 회차가 이미 주문 대상이 아니면(보유 상태 변경) ❌.
REACTION_PENDING = "hourglass_flowing_sand"
REACTION_DONE = "white_check_mark"
REACTION_EXPIRED = "x"
_REACTION_NOOP_ERRORS = ("already_reacted", "no_reaction")


def _env_word(env: str) -> str:
    """메시지에 쓰는 계좌 구분."""
    return "모의" if env == ACCOUNT_ENV_MOCK else "운영"


def _require_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise BrokerApiError(f"{name} 환경변수가 필요합니다.")
    return value


def _round_text(round_no: int | None) -> str:
    """신호 주문의 회차 표기 — 직접 낸 주문(회차 없음)은 비운다."""
    return f"{round_no}호 " if round_no is not None else ""


def _is_swap(request: dict[str, Any]) -> bool:
    """매도 후 재매수 한 짝의 요청인지 — 이 기능 전에 보낸 메시지의 요청에는 키가 없고, 그건 단독 주문이다."""
    return bool(request.get("rebuy"))


def _order_text(request: dict[str, Any]) -> str:
    if _is_swap(request):
        sell_quantity = request.get("sell_quantity")
        buy_quantity = request.get("buy_quantity")
        sell = f"{sell_quantity:,}주 매도" if sell_quantity is not None else "매도"
        buy = f"{buy_quantity:,}주 재매수" if buy_quantity is not None else "재매수"
        return f"{_round_text(request['round'])}{request['label']} · {sell} 후 {buy}"
    side = _action_word(request)
    return f"{_round_text(request['round'])}{request['label']} · {request['quantity']:,}주 × {_QUOTE_LABELS[request['side']]} 지정가 {side}"


def _action_word(request: dict[str, Any]) -> str:
    """메시지·모달 제목의 주문 구분 — 매수·매도, 또는 매도 후 재매수."""
    if request.get("arbitrary_id"):
        return "임의 매도"
    return "매도 후 재매수" if _is_swap(request) else _SIDE_LABELS[request["side"]]


# --- 메시지 ---------------------------------------------------------------------------------------------


def _section(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*[{_env_word(request['env'])}] {_action_word(request)} 주문*\n{_order_text(request)}\n"
                + (
                    f"매도 1호가에 매도, 매수 1호가에 재매수 주문을 함께 겁니다 ({request['ticker']})"
                    if _is_swap(request)
                    else f"버튼을 확정하는 순간의 {_QUOTE_LABELS[request['side']]}로 냅니다 ({request['ticker']})"
                )
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
    style = "primary" if request["side"] == "buy" and not _is_swap(request) else "danger"
    return [_section(request), _button(OPEN_ACTION_ID, f"{_action_word(request)} {label}", style, {"request": request})]


def _cancel_blocks(request: dict[str, Any], order_id: str) -> list[dict[str, Any]]:
    side = _SIDE_LABELS[request["side"]]
    return [
        _section(request),
        _button(CANCEL_ACTION_ID, f"{side} 주문 취소하기", "danger", {"request": request, "order_id": order_id}),
    ]


def _note_blocks(request: dict[str, Any], note: str) -> list[dict[str, Any]]:
    return [_section(request), _note(note)]


def _pair_blocks(request: dict[str, Any], pair_id: str, note: str | None) -> list[dict[str, Any]]:
    """매도 후 재매수 메시지 — 아직 끝나지 않은 쪽마다 취소 버튼을 단다(한쪽이 끝나도 다른 쪽 버튼은 남는다)."""
    records = list(pair_records(pair_id))
    request = {**request, **{f"{record['side']}_quantity": record["quantity"] for record in records}}
    blocks = [_section(request)]
    for record in records:
        order = serialize_order(record)
        if order["status"] in FINISHED_STATUSES:
            continue
        single = {**request, "side": order["side"], "quantity": order["quantity"], "rebuy": False}
        blocks.append(
            _button(
                CANCEL_ACTION_ID,
                f"{_SIDE_LABELS[order['side']]} 주문 취소하기",
                "danger",
                {"request": single, "order_id": order["id"]},
            )
        )
    if note:
        blocks.append(_note(note))
    return blocks


def _update_message(target: dict[str, Any], blocks: list[dict[str, Any]]) -> None:
    """원 메시지의 버튼·상태 줄을 교체한다."""
    WebClient(token=_require_env("SLACK_BOT_TOKEN")).chat_update(
        channel=target["channel"],
        ts=target["message_ts"],
        text=f"[{_env_word(target['request']['env'])}] {_order_text(target['request'])}",
        blocks=blocks,
    )


def send_order_message(
    account_id: str,
    pool: str,
    strategy_id: str,
    side: str,
    round_no: int | None,
    thread_ts: str | None,
    *,
    rebuy: bool,
    arbitrary_id: str | None = None,
) -> None:
    """주문 메시지를 슬랙으로 보낸다(thread_ts 가 있으면 그 스레드에) — 주문은 슬랙에서 버튼과 모달 확인을 거쳐야 나간다.

    ``rebuy`` 는 매도 후 재매수 한 짝의 메시지다(회차 매도 신호 중 팔자마자 다시 사게 되는 경우).
    """
    env = account_env(account_id)
    if rebuy:
        if side != "sell" or round_no is None:
            raise BrokerApiError("매도 후 재매수는 회차 매도 신호에만 보냅니다.")
        order = resolve_swap(account_id, pool, strategy_id, round_no, fresh=False)
    else:
        order = resolve_order(account_id, pool, strategy_id, side, round_no, fresh=False, arbitrary_id=arbitrary_id)
    request = {
        "env": env,
        "round": round_no,
        "account_id": account_id,
        "pool": pool,
        "strategy_id": strategy_id,
        "side": side,
        "label": order["strategy_label"],
        "ticker": order["ticker"],
        **(
            {key: order[key] for key in ("sell_quantity", "buy_quantity")} if rebuy else {"quantity": order["quantity"]}
        ),
        "rebuy": rebuy,
        "arbitrary_id": arbitrary_id,
    }
    if not send_thread_message(f"[{_env_word(env)}] {_order_text(request)}", _open_blocks(request), thread_ts):
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


def _account_line(request: dict[str, Any]) -> str:
    """모달에 보일 주문 계좌 — 계좌 설정의 이름과 ID."""
    from utils.account_settings_store import load_account_docs

    doc = next((doc for doc in load_account_docs() if doc["account_id"] == request["account_id"]), None)
    if doc is None:
        raise BrokerApiError(f"'{request['account_id']}' 계좌가 계좌 설정에 없습니다.")
    return f"계좌: {doc.get('name') or doc['account_id']}({doc['account_id']})"


def _signal_gap_pct(order: dict[str, Any]) -> float | None:
    """신호 가격(매도 목표·매수선) 대비 주문 가격 차이 % — 신호 가격이 없는 주문(장 마감 매수)은 None."""
    if order["signal_price"] is None:
        return None
    return (order["price"] / order["signal_price"] - 1.0) * 100.0


def _gap_label(side: str, gap_pct: float) -> str:
    """신호 대비 차이를 사람이 바로 판단하게 적는다 — 매도는 낮을수록, 매수는 높을수록 신호보다 불리하다."""
    unfavorable = gap_pct < 0 if side == "sell" else gap_pct > 0
    return "⚠️ 신호보다 불리" if unfavorable else "신호보다 유리"


def _order_view(request: dict[str, Any], order: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    side = _action_word(request)
    gap_pct = _signal_gap_pct(order)
    if gap_pct is None:
        gap = f"{order['quote_label']} {order['price']:,}원 (신호 가격 비교 없음)"
    else:
        gap = (
            f"*신호 대비 {gap_pct:+.2f}%* — {_gap_label(request['side'], gap_pct)}\n"
            f"신호 {order['signal_price']:,.0f}원 · {order['quote_label']} {order['price']:,}원"
        )
    guide = (
        f"주문을 확정하는 순간의 {order['quote_label']}로 {_env_word(request['env'])} 계좌에 보냅니다. "
        "호가가 움직이면 가격과 매수 수량이 달라질 수 있습니다."
    )
    body = f"*{_order_text({**request, 'quantity': order['quantity']})}*\n{_account_line(request)}\n{gap}\n{guide}"
    return _modal_view(ORDER_CALLBACK_ID, f"{_env_word(request['env'])} {side} 주문 확인", f"{side} 주문", body, target)


def _swap_view(request: dict[str, Any], order: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    signal_gap = (order["sell_price"] / order["signal_price"] - 1.0) * 100.0
    body = (
        f"*{_order_text({**request, **order})}*\n{_account_line(request)}\n"
        f"매도 {order['sell_price']:,}원(매도 1호가) · 매수 {order['buy_price']:,}원(매수 1호가)\n"
        f"매도 신호 {order['signal_price']:,.0f}원 대비 {signal_gap:+.2f}%\n"
        f"확정하는 순간의 호가로 {_env_word(request['env'])} 계좌에 매도·매수 지정가를 함께 겁니다. "
        "호가가 움직이면 가격과 재매수 수량이 달라질 수 있습니다."
    )
    return _modal_view(
        ORDER_CALLBACK_ID, f"{_env_word(request['env'])} 매도 후 재매수 확인", "매도·매수 주문", body, target
    )


def _cancel_view(request: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    side = _SIDE_LABELS[request["side"]]
    body = f"*{_order_text(request)}*\n{_account_line(request)}\n접수된 {_env_word(request['env'])} {side} 주문을 취소합니다(일부 체결됐다면 남은 수량만 취소됩니다)."
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
    request = {**value["request"], "env": account_env(value["request"]["account_id"])}
    message = payload["message"]
    target = {
        "request": request,
        "channel": payload["channel"]["id"],
        "message_ts": message["ts"],
        # 스레드 답글이면 최상위 메시지의 ts, 최상위 메시지면 자기 ts 가 스레드의 뿌리다.
        "thread_ts": message.get("thread_ts") or message["ts"],
    }
    if action["action_id"] == OPEN_ACTION_ID:
        try:
            # 슬랙 trigger_id 는 3초 안에 써야 해서 저장된 원장으로 연다 — 실제 주문 직전 검증은 `place_order` 가 새로 계산한다.
            if _is_swap(request):
                order = resolve_swap(
                    request["account_id"], request["pool"], request["strategy_id"], request["round"], fresh=False
                )
            else:
                order = resolve_order(
                    request["account_id"],
                    request["pool"],
                    request["strategy_id"],
                    request["side"],
                    request["round"],
                    fresh=False,
                    arbitrary_id=request.get("arbitrary_id"),
                )
        except RoundNotOrderableError as exc:
            _expire(target, exc)
            return
        except BrokerApiError as exc:
            # 다시 누르면 될 수 있는 실패(수동 미체결 주문·호가 없음·연동 오류 등) — 버튼은 두고 이유만 스레드에 남긴다.
            _report_open_failure(target, exc)
            return
        view = _swap_view(request, order, target) if _is_swap(request) else _order_view(request, order, target)
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
    side = f"{_env_word(order['env'])} {_action_word(order)}"
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
    send_thread_message(text, thread_ts=target["thread_ts"])
    request = target["request"]
    if order["pair_id"] is not None:
        _update_message(target, _pair_blocks(request, order["pair_id"], text))
        return
    if order["status"] == "canceled" and order["filled_quantity"] == 0:
        # 다시 주문해도 서버가 회차 보유 상태·중복·안전 장치를 처음 주문과 똑같이 검사한다.
        _update_message(target, [*_open_blocks(request, "다시 주문하기"), _note(text)])
    else:
        _update_message(target, _note_blocks(request, text))


def _report_open_failure(target: dict[str, Any], exc: Exception) -> None:
    """버튼을 눌렀는데 확인 모달을 열 수 없을 때 — 이유를 스레드에 남긴다(응답 없이 끝나지 않게)."""
    request = target["request"]
    side = f"{_env_word(request['env'])} {_action_word(request)}"
    send_thread_message(f"❌ {side} 주문 확인을 열 수 없습니다: {exc}", thread_ts=target["thread_ts"])


def _expire(target: dict[str, Any], exc: Exception) -> None:
    """회차가 더는 주문 대상이 아니면 다시 눌러도 같은 결과이므로 버튼을 되살리지 않고 신호 메시지에 ❌ 를 단다."""
    request = target["request"]
    side = f"{_env_word(request['env'])} {_action_word(request)}"
    send_thread_message(f"❌ {side} 주문 실패: {exc}", thread_ts=target["thread_ts"])
    _update_message(target, _note_blocks(request, "⌛ 보유 상태가 바뀌어 주문 불가"))
    if target["thread_ts"] != target["message_ts"]:
        _react("reactions_add", target["channel"], target["thread_ts"], REACTION_EXPIRED)


def _place_swap(target: dict[str, Any]) -> None:
    """매도 후 재매수 한 짝을 낸다 — 결과를 스레드로 알리고 메시지에 양쪽 취소 버튼을 단다."""
    request = target["request"]
    word = f"{_env_word(request['env'])} 매도 후 재매수"
    _update_message(target, _note_blocks(request, "⏳ 주문 처리 중…"))
    try:
        result = place_swap(
            request["account_id"],
            request["pool"],
            request["strategy_id"],
            request["round"],
            {key: target[key] for key in SLACK_TARGET_KEYS},
        )
    except RoundNotOrderableError as exc:
        _expire(target, exc)
        return
    except Exception as exc:  # 슬랙 요청은 이미 응답했으므로 실패도 스레드로 알린다.
        logger.exception("슬랙 매도 후 재매수 처리 실패")
        send_thread_message(f"❌ {word} 주문 실패: {exc}", thread_ts=target["thread_ts"])
        _update_message(target, _open_blocks(request))
        return
    sell, buy = result["sell"], result["buy"]
    text = f"✅ {word} 접수: {sell['strategy_label']} — 매도 {sell['quantity']:,}주 × {sell['price']:,}원"
    if buy is not None:
        text += f" · 매수 {buy['quantity']:,}주 × {buy['price']:,}원"
    else:
        text += f"\n⚠️ 매수 주문은 실패했습니다(매도만 걸림): {result['buy_error']}"
    send_thread_message(text, thread_ts=target["thread_ts"])
    _update_message(target, _pair_blocks(request, sell["pair_id"], None))
    if target["thread_ts"] != target["message_ts"]:
        _react("reactions_add", target["channel"], target["thread_ts"], REACTION_PENDING)


def _place(target: dict[str, Any]) -> None:
    request = target["request"]
    if _is_swap(request):
        _place_swap(target)
        return
    side = f"{_env_word(request['env'])} {_action_word(request)}"
    # 처리하는 동안 같은 메시지로 또 주문하지 못하게 버튼부터 걷어낸다.
    _update_message(target, _note_blocks(request, "⏳ 주문 처리 중…"))
    try:
        order = place_order(
            request["account_id"],
            request["pool"],
            request["strategy_id"],
            request["side"],
            request["round"],
            {key: target[key] for key in SLACK_TARGET_KEYS},
            arbitrary_id=request.get("arbitrary_id"),
        )
    except RoundNotOrderableError as exc:
        _expire(target, exc)
        return
    except Exception as exc:  # 슬랙 요청은 이미 응답했으므로 실패도 스레드로 알린다.
        logger.exception("슬랙 모의 주문 처리 실패")
        send_thread_message(f"❌ {side} 주문 실패: {exc}", thread_ts=target["thread_ts"])
        _update_message(target, _open_blocks(request))
        return
    gap_pct = _signal_gap_pct(order)
    gap = f" · 신호 대비 {gap_pct:+.2f}%" if gap_pct is not None else ""
    send_thread_message(
        f"✅ {side} 주문 접수: {order['strategy_label']} {order['quantity']:,}주 × {order['price']:,}원 (지정가){gap}",
        thread_ts=target["thread_ts"],
    )
    _update_message(target, _cancel_blocks(request, order["id"]))
    # 신호 스레드의 첫 주문이 접수되면 신호 메시지에 모래시계를 단다(이미 달려 있으면 무시된다).
    if target["thread_ts"] != target["message_ts"]:
        _react("reactions_add", target["channel"], target["thread_ts"], REACTION_PENDING)


def _cancel(target: dict[str, Any]) -> None:
    request = target["request"]
    side = f"{_env_word(request['env'])} {_SIDE_LABELS[request['side']]}"
    _update_message(target, _note_blocks(request, "⏳ 취소 처리 중…"))
    try:
        cancel_record(target["order_id"])
    except Exception as exc:
        logger.exception("슬랙 모의 주문 취소 실패")
        send_thread_message(f"❌ {side} 주문 취소 실패: {exc}", thread_ts=target["thread_ts"])
        # 그 사이 주문이 끝났을 수 있다 — 끝났으면 결과 알림은 배치가 하고, 아니면 취소 버튼을 되살린다.
        order = refreshed_order(request["account_id"], target["order_id"])
        if order["pair_id"] is not None:
            # 매도 후 재매수 메시지 — 다른 쪽 취소 버튼까지 다시 그린다.
            swap = {**request, "side": "sell", "rebuy": True}
            _update_message({**target, "request": swap}, _pair_blocks(swap, order["pair_id"], None))
        elif order["status"] in FINISHED_STATUSES:
            _update_message(target, _note_blocks(request, "⏳ 결과 확인 중…"))
        else:
            _update_message(target, _cancel_blocks(request, target["order_id"]))
        return
    send_thread_message(f"🚫 {side} 주문 취소를 요청했습니다.", thread_ts=target["thread_ts"])
    # 취소 결과(체결 수량 포함)로 메시지를 마무리하는 일은 배치(`notify_finished_orders`)가 한다.


def run_submission(target: dict[str, Any]) -> None:
    """확인된 주문·취소를 처리한다 — 응답 뒤 백그라운드에서 돈다."""
    try:
        if target["kind"] == "order":
            from utils.strategy_trade_threads import reserve_signal_for_order

            if not reserve_signal_for_order(target["thread_ts"]):
                return
            _place(target)
        else:
            _cancel(target)
    except Exception as exc:
        logger.exception("슬랙 모의 주문 후속 처리 실패")
        send_thread_message(f"⚠️ 처리 중 오류로 멈춥니다: {exc}", thread_ts=target["thread_ts"])


def _request_of(record: dict[str, Any]) -> dict[str, Any]:
    """주문 기록으로 메시지에 그릴 주문 내용을 만든다 — 매도 후 재매수 한 짝이면 그 메시지(매도·재매수)의 내용이다."""
    swap = record.get("pair_id") is not None
    return {
        "env": record["env"],
        "round": record["round"],
        "account_id": record["account_id"],
        "pool": record["pool"],
        "strategy_id": record["strategy_id"],
        "side": "sell" if swap else record["side"],
        "label": record["strategy_label"],
        "ticker": record["ticker"],
        "quantity": record["quantity"],
        "rebuy": swap,
        "arbitrary_id": record.get("arbitrary_id"),
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
        close_at = venue_close_time(record["venue"])
        text = f"⏰ {side} 주문이 아직 체결되지 않았습니다(미체결 {remaining:,}주). {close_at} {record['venue']} 마감 전에 취소하지 않으면 오늘 주문은 만료됩니다."
        send_thread_message(text, thread_ts=target["thread_ts"])
        if order["pair_id"] is not None:
            _update_message(target, _pair_blocks(request, order["pair_id"], text))
        else:
            _update_message(target, [*_cancel_blocks(request, order["id"]), _note(text)])
        mark_slack_close_warned(record["_id"])
    return len(records)


def _react(method: str, channel: str, ts: str, name: str) -> None:
    """메시지 반응을 달거나 뗀다 — 반응은 부가 표시라 실패해도 로그만 남기고 흐름을 막지 않는다(권한: reactions:write)."""
    client = WebClient(token=_require_env("SLACK_BOT_TOKEN"))
    try:
        getattr(client, method)(channel=channel, timestamp=ts, name=name)
    except SlackApiError as exc:
        error = exc.response.get("error")
        if error not in _REACTION_NOOP_ERRORS:
            logger.error("슬랙 반응 %s(%s) 실패: %s", method, name, error)


def notify_completed_threads() -> int:
    """주문 메시지가 모두 체결된 신호 스레드의 신호 메시지에서 모래시계를 떼고 완료(✅)를 단다 — 완료한 스레드 수."""
    threads = filled_signal_threads()
    for thread in threads:
        _react("reactions_remove", thread["channel"], thread["thread_ts"], REACTION_PENDING)
        _react("reactions_add", thread["channel"], thread["thread_ts"], REACTION_DONE)
        mark_signal_thread_completed(thread["thread_ts"])
    return len(threads)


def send_signal_order_messages(strategy_view: dict[str, Any], triggers: list[dict[str, Any]], signal_ts: str) -> int:
    """신호가 난 회차마다 주문 메시지(버튼)를 신호 메시지의 스레드로 보내고 그 스레드를 추적 대상으로 기록한다 — 보낸 건수.

    이미 주문이 진행 중인 회차는 건너뛴다(매시간 재알림에서 주문이 아직 없는 회차에만 다시 띄우려는 것).
    주문은 슬랙의 버튼과 모달 확인을 거쳐야 나간다.
    """
    account_id = strategy_view["account_id"]
    pending = [
        trigger
        for trigger in triggers
        if find_duplicate_order(account_id, trigger["strategy_id"], trigger["action"]) is None
        # 매도 후 재매수는 같은 회차의 매수 주문도 걸리므로 그쪽 중복도 본다.
        and not (trigger["rebuy"] and find_duplicate_order(account_id, trigger["strategy_id"], "buy") is not None)
    ]
    sent = 0
    try:
        for trigger in pending:
            send_order_message(
                account_id,
                strategy_view["pool"],
                trigger["strategy_id"],
                trigger["action"],
                trigger["round"],
                signal_ts,
                rebuy=trigger["rebuy"],
                arbitrary_id=trigger.get("arbitrary_id"),
            )
            sent += 1
    finally:
        # 중간에 실패해도 이미 보낸 메시지는 스레드로 추적한다.
        if sent:
            channel = get_slack_channel()
            if register_signal_thread(channel, signal_ts, sent):
                _react("reactions_remove", channel, signal_ts, REACTION_DONE)
    return sent
