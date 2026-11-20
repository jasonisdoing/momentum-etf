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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qs

from bson import ObjectId
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier

from services.broker_api_service import BrokerApiError, OrderUncertainError
from services.broker_trade_service import account_env
from utils.account_settings_store import ACCOUNT_ENV_MOCK
from utils.notification import get_slack_client
from utils.settings_loader import get_slack_channel
from utils.strategy_trade_market import format_money
from utils.strategy_trade_orders import (
    ACTIVE_STATUSES,
    FINISHED_STATUSES,
    MESSAGE_DONE,
    MESSAGE_FAILED,
    MESSAGE_OPEN,
    MESSAGE_PENDING,
    MESSAGE_PLACED,
    RoundNotOrderableError,
    _orders,
    cancel_record,
    close_warning_slack_orders,
    close_warning_text,
    find_duplicate_order,
    find_signal_thread,
    finished_slack_orders,
    forget_signal_thread,
    mark_slack_close_warned,
    mark_slack_notified,
    modify_record,
    pair_records,
    place_order,
    place_swap,
    preview_modify_record,
    refresh_slack_order_accounts,
    refreshed_order,
    register_signal_thread,
    resolve_order,
    resolve_swap,
    round_done,
    serialize_order,
    set_signal_thread_reaction,
    settle_signal_message,
    signal_message_states,
    signal_thread_state,
    signal_threads,
)
from utils.strategy_trade_signal_lock import signal_lock
from utils.strategy_trade_threads import (
    can_withdraw_signal,
    finish_order_reservation,
    reserve_signal_for_order,
    send_thread_message,
)

logger = logging.getLogger(__name__)

OPEN_ACTION_ID = "mock_order_open"
MODIFY_ACTION_ID = "mock_order_modify"
CANCEL_ACTION_ID = "mock_order_cancel"
ORDER_CALLBACK_ID = "mock_order_confirm"
MODIFY_CALLBACK_ID = "mock_order_modify_confirm"
CANCEL_CALLBACK_ID = "mock_order_cancel_confirm"
_SIDE_LABELS = {"buy": "매수", "sell": "매도"}
#: 주문 가격으로 쓰는 같은 쪽 1호가 — 매수는 매수 1호가, 매도는 매도 1호가에 맞춘다.
_QUOTE_LABELS = {"buy": "매수 1호가", "sell": "매도 1호가"}
#: 주문 기록(`slack`)에 남기는 슬랙 위치 — 버튼이 있는 메시지(`message_ts`)와 결과를 달 스레드 뿌리(`thread_ts`).
SLACK_TARGET_KEYS = ("channel", "message_ts", "thread_ts")
#: 신호 메시지 반응 — 스레드 상태(`signal_thread_state`)마다 하나만 단다: 처리 중 ⏳ · 접수됨 🎯 · 완료 ✅ · 실패 ❌.
REACTIONS = {
    MESSAGE_PENDING: "hourglass_flowing_sand",
    MESSAGE_PLACED: "dart",
    MESSAGE_DONE: "white_check_mark",
    MESSAGE_FAILED: "x",
}
_REACTION_NOOP_ERRORS = ("already_reacted", "no_reaction")
#: 장이 닫혀 주문하지 못한 메시지에 남기는 문구.
_SESSION_CLOSED_NOTE = "⌛ 장 마감으로 주문 불가"
#: 다른 주문으로 이미 처리된 회차의 메시지에 남기는 문구.
_ALREADY_DONE_NOTE = "✅ 이미 처리된 회차라 주문하지 않음"


#: 자동 주문 표시 — 자동 신호 머리말·주문 메시지·결과 답글 앞에 붙여 수동과 구분한다.
AUTO_ORDER_TAG = "🤖[자동]"


def is_auto_trigger(strategy_view: dict[str, Any], trigger: dict[str, Any]) -> bool:
    """자동 주문 대상 신호인지 — 「자동」을 켠 전략의 자동 신호만. 화면에서 보낸 수동 신호는 버튼으로 둔다."""
    return bool(strategy_view["config"]["auto_order"]) and trigger.get("origin") != "manual"


def _slack_of(target: dict[str, Any]) -> dict[str, Any]:
    """주문 기록에 남길 슬랙 위치 — 자동 주문 여부도 함께 남겨 끝난 뒤 결과 답글에도 표시한다."""
    return {**{key: target[key] for key in SLACK_TARGET_KEYS}, "auto": target["request"].get("auto") is True}


def _auto_tag(request: dict[str, Any]) -> str:
    # 자동 주문 표시가 생기기 전 버튼·주문 기록에는 키가 없다 — 그것들은 모두 수동 주문이다.
    return f"{AUTO_ORDER_TAG} " if request.get("auto") is True else ""


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
    if request.get("mix_action_key") is not None:
        return f"{request['label']} · {request['quantity']:,}주 {side}"
    return f"{_round_text(request['round'])}{request['label']} · {request['quantity']:,}주 × {_QUOTE_LABELS[request['side']]} 지정가 {side}"


def _action_word(request: dict[str, Any]) -> str:
    """메시지·모달 제목의 주문 구분 — 매수·매도, 또는 매도 후 재매수."""
    if request.get("arbitrary_id"):
        return "임의 매도"
    return "매도 후 재매수" if _is_swap(request) else _SIDE_LABELS[request["side"]]


# --- 메시지 ---------------------------------------------------------------------------------------------


def _section(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("mix_action_key") is not None:
        detail = "확인창에서 주문 지정가를 확인합니다"
    elif _is_swap(request):
        detail = "매도 1호가에 매도, 매수 1호가에 재매수 주문을 함께 겁니다"
    elif request.get("auto") is True:
        detail = f"신호와 함께 자동으로 그 순간의 {_QUOTE_LABELS[request['side']]}로 냅니다"
    else:
        detail = f"버튼을 확정하는 순간의 {_QUOTE_LABELS[request['side']]}로 냅니다"
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*{_auto_tag(request)}[{_env_word(request['env'])}] {_action_word(request)} 주문{_venue_text(request)}*\n"
                f"{_order_text(request)}\n"
                f"{detail} ({request['ticker']})"
            ),
        },
    }


def _venue_text(request: dict[str, Any]) -> str:
    # 거래소는 주문을 낸 뒤에야 정해진다(주문 순간의 시간대로 고른다) — 낸 주문의 메시지에만 붙인다.
    return f"({request['venue']})" if request.get("venue") else ""


def _with_venue(request: dict[str, Any], venue: str) -> dict[str, Any]:
    """낸 주문의 거래소를 실은 주문 내용 — 이후 정정·취소 버튼과 메시지 제목이 이 값을 쓴다."""
    return {**request, "venue": venue}


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
    # 새로 낼 주문은 누르는 순간의 거래소로 나가므로 앞 주문의 거래소를 지운다.
    request = {key: value for key, value in request.items() if key != "venue"}
    style = "primary" if request["side"] == "buy" and not _is_swap(request) else "danger"
    return [_section(request), _button(OPEN_ACTION_ID, f"{_action_word(request)} {label}", style, {"request": request})]


def _order_action_buttons(request: dict[str, Any], order_id: str, prefix: str = "") -> dict[str, Any]:
    return {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "action_id": MODIFY_ACTION_ID,
                "text": {"type": "plain_text", "text": f"{prefix}호가 정정"},
                "style": "primary",
                "value": json.dumps({"request": request, "order_id": order_id}, ensure_ascii=False),
            },
            {
                "type": "button",
                "action_id": CANCEL_ACTION_ID,
                "text": {"type": "plain_text", "text": f"{prefix}주문 취소하기"},
                "style": "danger",
                "value": json.dumps({"request": request, "order_id": order_id}, ensure_ascii=False),
            },
        ],
    }


def _cancel_blocks(request: dict[str, Any], order_id: str) -> list[dict[str, Any]]:
    side = _SIDE_LABELS[request["side"]]
    return [
        _section(request),
        _order_action_buttons(request, order_id, f"{side} "),
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
        blocks.append(_order_action_buttons(single, order["id"], f"{_SIDE_LABELS[order['side']]} "))
    if note:
        blocks.append(_note(note))
    return blocks


def _slack_client():
    return get_slack_client(_require_env("SLACK_BOT_TOKEN"))


def _update_message(target: dict[str, Any], blocks: list[dict[str, Any]]) -> None:
    """원 메시지의 버튼·상태 줄을 교체한다."""
    _slack_client().chat_update(
        channel=target["channel"],
        ts=target["message_ts"],
        text=f"{_auto_tag(target['request'])}[{_env_word(target['request']['env'])}] {_order_text(target['request'])}",
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
    auto: bool,
    arbitrary_id: str | None = None,
) -> dict[str, Any]:
    """주문 메시지를 슬랙으로 보낸다(thread_ts 가 있으면 그 스레드에) — 주문은 슬랙에서 버튼과 모달 확인을 거쳐야 나간다.

    ``rebuy`` 는 매도 후 재매수 한 짝의 메시지다(회차 매도 신호 중 팔자마자 다시 사게 되는 경우).
    보낸 메시지의 ``{"ts", "request"}`` 를 돌려준다(신호 스레드 기록이 버튼을 다시 그릴 때 쓴다).
    """
    if rebuy:
        if side != "sell" or round_no is None:
            raise BrokerApiError("매도 후 재매수는 회차 매도 신호에만 보냅니다.")
        order = resolve_swap(account_id, pool, strategy_id, round_no, fresh=False)
    else:
        order = resolve_order(account_id, pool, strategy_id, side, round_no, fresh=False, arbitrary_id=arbitrary_id)
    return _send_order_request(
        account_id, pool, strategy_id, side, round_no, thread_ts, rebuy, arbitrary_id, order, auto=auto
    )


def _send_order_request(
    account_id: str,
    pool: str,
    strategy_id: str,
    side: str,
    round_no: int | None,
    thread_ts: str | None,
    rebuy: bool,
    arbitrary_id: str | None,
    order: dict[str, Any],
    *,
    auto: bool,
) -> dict[str, Any]:
    """확정된 표시 내용으로 기존 주문 버튼을 만든다. 실제 호가는 Slack 확인(자동이면 바로 이어지는 주문) 시 조회한다."""
    env = account_env(account_id)
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
        "auto": auto,
    }
    return send_order_request(request, thread_ts)


def send_order_request(request: dict[str, Any], thread_ts: str | None) -> dict[str, Any]:
    """전략에서 구성한 주문 요청을 공통 Slack 버튼으로 보낸다."""
    ts = send_thread_message(
        f"{_auto_tag(request)}[{_env_word(request['env'])}] {_order_text(request)}", _open_blocks(request), thread_ts
    )
    if not ts:
        raise BrokerApiError("슬랙으로 주문 메시지를 보내지 못했습니다.")
    return {"ts": ts, "request": request}


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
    """신호 가격(매도 목표·매수선) 대비 주문 가격 차이 % — 신호 가격이 없는 주문(정해진 시각에 사는 1호)은 None."""
    if order["signal_price"] is None:
        return None
    return (order["price"] / order["signal_price"] - 1.0) * 100.0


def _gap_label(side: str, gap_pct: float) -> str:
    """신호 대비 차이를 사람이 바로 판단하게 적는다 — 매도는 낮을수록, 매수는 높을수록 신호보다 불리하다."""
    unfavorable = gap_pct < 0 if side == "sell" else gap_pct > 0
    return "⚠️ 신호보다 불리" if unfavorable else "신호보다 유리"


#: 확인 모달에 보이는 호가 단계 수 — 매도·매수 각각.
QUOTE_DEPTH_SHOWN = 3


def _quote_depth_text(order: dict[str, Any], marks: dict[tuple[str, int], str]) -> str:
    """확인 모달의 호가 표 — 매도 3호가→1호가, 매수 1호가→3호가와 전일 종가 대비 %·잔량. ``marks`` 는 (asks·bids, 단계) → 주문 표시.

    증권사 연동이 단계 호가를 주지 않으면(나무) 그 사실을 한 줄로 보인다.
    """
    depth = order["quote_depth"]
    if depth is None:
        return "호가 단계: 이 증권사 연동은 단계 호가를 받지 않습니다."
    prev_close = depth["prev_close"]
    lines = []
    for side, word, levels in (
        ("asks", "매도", list(enumerate(depth["asks"][:QUOTE_DEPTH_SHOWN], 1))[::-1]),
        ("bids", "매수", list(enumerate(depth["bids"][:QUOTE_DEPTH_SHOWN], 1))),
    ):
        for level, row in levels:
            mark = marks.get((side, level), "")
            price = format_money(row["price"], order["country"])
            change = f"{(row['price'] / prev_close - 1) * 100:+.2f}%" if prev_close else "-"
            lines.append(
                f"{word}{level} {price:>11} {change:>8} {row['quantity']:>9,}주{f'  ← {mark}' if mark else ''}"
            )
    basis = (
        f"전일 종가 {format_money(prev_close, order['country'])} 대비" if prev_close else "전일 종가를 받지 못해 % 없음"
    )
    return f"{basis}\n```" + "\n".join(lines) + "```"


def _profit_summary_line(request: dict[str, Any], order: dict[str, Any]) -> str | None:
    """사고팔기 주문 확인 모달용 종목·회차 수익률 한 줄 요약 — 합성은 None."""
    if request.get("mix_action_key") is not None:
        return None
    parts: list[str] = []
    if order.get("stock_profit_pct") is not None and order.get("average_price") is not None:
        parts.append(
            f"종목 수익률 *{order['stock_profit_pct']:+.2f}%* (평단 {format_money(order['average_price'], order['country'])})"
        )
    if order.get("round_profit_pct") is not None and order.get("round_fill_price") is not None:
        round_label = f"{request['round']}호 수익률" if request.get("round") is not None else "회차 수익률"
        parts.append(
            f"{round_label} *{order['round_profit_pct']:+.2f}%* (체결 {format_money(order['round_fill_price'], order['country'])})"
        )
    return " · ".join(parts) if parts else None


def _order_price_details(request: dict[str, Any], order: dict[str, Any]) -> str:
    price = format_money(order["price"], order["country"])
    # 매수는 매수 1호가, 매도는 매도 1호가로 낸다 — 표에서 그 줄을 짚는다.
    depth = _quote_depth_text(order, {("bids", 1) if request["side"] == "buy" else ("asks", 1): "주문 가격"})
    if request.get("mix_action_key") is not None:
        return f"지정가 {price}\n{depth}\n확인한 수량이나 지정가가 바뀌면 다시 확인해야 합니다."
    gap_pct = _signal_gap_pct(order)
    if gap_pct is None:
        gap = f"{order['quote_label']} {price} (신호 가격 비교 없음)"
    else:
        gap = (
            f"*신호 대비 {gap_pct:+.2f}%* — {_gap_label(request['side'], gap_pct)}\n"
            f"신호 {format_money(order['signal_price'], order['country'])} · {order['quote_label']} {price}"
        )
    profit_line = _profit_summary_line(request, order)
    guide = (
        f"주문을 확정하는 순간의 {order['quote_label']}로 {_env_word(request['env'])} 계좌에 보냅니다. "
        "호가가 움직이면 가격과 매수 수량이 달라질 수 있습니다."
    )
    sections = [gap, *([profit_line] if profit_line else []), depth, guide]
    return "\n".join(sections)


def _order_view(request: dict[str, Any], order: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    side = _action_word(request)
    summary = _order_text({**request, "quantity": order["quantity"]})
    body = f"*{summary}*\n{_account_line(request)}\n{_order_price_details(request, order)}"
    return _modal_view(ORDER_CALLBACK_ID, f"{_env_word(request['env'])} {side} 주문 확인", f"{side} 주문", body, target)


def _swap_view(request: dict[str, Any], order: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    signal_gap = (order["sell_price"] / order["signal_price"] - 1.0) * 100.0
    profit_line = _profit_summary_line(request, order)
    lines = [
        f"*{_order_text({**request, **order})}*",
        _account_line(request),
        f"매도 {format_money(order['sell_price'], order['country'])}(매도 1호가) · 매수 {format_money(order['buy_price'], order['country'])}(매수 1호가)",
        f"매도 신호 {format_money(order['signal_price'], order['country'])} 대비 {signal_gap:+.2f}%",
        *([profit_line] if profit_line else []),
        _quote_depth_text(order, {("asks", 1): "매도", ("bids", 1): "재매수"}),
        f"확정하는 순간의 호가로 {_env_word(request['env'])} 계좌에 매도·매수 지정가를 함께 겁니다. "
        "호가가 움직이면 가격과 재매수 수량이 달라질 수 있습니다.",
    ]
    return _modal_view(
        ORDER_CALLBACK_ID,
        f"{_env_word(request['env'])} 매도 후 재매수 확인",
        "매도·매수 주문",
        "\n".join(lines),
        target,
    )


def _modify_view(request: dict[str, Any], preview: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    side = _SIDE_LABELS[request["side"]]
    country = preview["order"]["country"]
    depth = _quote_depth_text(
        {"quote_depth": preview.get("quote_depth"), "country": country},
        {("bids", 1) if request["side"] == "buy" else ("asks", 1): "정정 가격"},
    )
    body = "\n".join(
        [
            f"*{_order_text(request)}*",
            _account_line(request),
            f"남은 수량: *{preview['remaining_quantity']:,}주*",
            f"현재 주문가: {format_money(preview['current_price'], country)}",
            f"정정 가격({preview['quote_label']}): *{format_money(preview['target_price'], country)}*",
            *([f"\n{depth}"] if depth else []),
            f"\n접수된 {_env_word(request['env'])} {side} 주문을 현재 상대 1호가로 정정합니다.",
        ]
    )
    return _modal_view(
        MODIFY_CALLBACK_ID, f"{_env_word(request['env'])} {side} 호가 정정 확인", "호가 정정", body, target
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


def _resolve_request(request: dict[str, Any], thread_ts: str, *, fresh: bool) -> dict[str, Any]:
    """주문 메시지의 주문 내용을 지금 보유 상태로 다시 계산한다 — 주문할 수 없는 회차면 `RoundNotOrderableError`."""
    if request.get("mix_action_key") is not None:
        from utils.strategy_mix_orders import resolve_mix_order

        return resolve_mix_order(request, thread_ts)
    if _is_swap(request):
        return resolve_swap(
            request["account_id"], request["pool"], request["strategy_id"], request["round"], fresh=fresh
        )
    return resolve_order(
        request["account_id"],
        request["pool"],
        request["strategy_id"],
        request["side"],
        request["round"],
        fresh=fresh,
        arbitrary_id=request.get("arbitrary_id"),
    )


def _status_view(request: dict[str, Any], target: dict[str, Any], note: str) -> dict[str, Any]:
    """제출 버튼 없는 확인 모달 — 버튼에 담긴 주문 내용과 상태 한 줄(확인 중·주문 불가·오류)만 보인다."""
    # 제목은 채워진 뒤의 확인 모달(`_order_view`·`_swap_view`)과 같다 — 열리고 나서 제목이 바뀌지 않게.
    env = _env_word(request["env"])
    title = f"{env} 매도 후 재매수 확인" if _is_swap(request) else f"{env} {_action_word(request)} 주문 확인"
    view = _modal_view(ORDER_CALLBACK_ID, title, "확인", f"*{_order_text(request)}*\n{note}", target)
    del view["submit"]
    return view


def _update_view(view_id: str, view: dict[str, Any]) -> None:
    """열린 모달을 바꾼다 — 사용자가 그 사이 모달을 닫았으면(not_found) 할 일이 없다."""
    try:
        _slack_client().views_update(view_id=view_id, view=view)
    except SlackApiError as exc:
        if exc.response.get("error") != "not_found":
            raise


def fill_confirm_modal(job: dict[str, Any]) -> None:
    """먼저 연 확인 모달에 호가·수량·신호 대비를 채운다 — 응답 뒤 백그라운드에서 돌아 3초 제한을 받지 않는다.

    저장된 원장으로 주문할 수 없다고 나오면(배치 주기만큼 늦을 수 있다) 새 원장으로 다시 판정하고, 그래도 주문할 수
    없으면 모달을 「주문 불가」로 바꾸고 버튼을 걷는다(`_expire`). 다른 실패는 이유를 모달에 보이고 버튼은 둔다.
    """
    target, view_id = job["target"], job["view_id"]
    request = target["request"]
    if job.get("kind") == "modify":
        try:
            preview = preview_modify_record(target["order_id"])
        except Exception as exc:
            logger.exception("슬랙 호가 정정 확인 모달 채우기 실패")
            _update_view(view_id, _status_view(request, target, f"❌ 호가 정정을 열 수 없습니다: {exc}"))
            return
        _update_view(view_id, _modify_view(request, preview, target))
        return
    try:
        try:
            order = _resolve_request(request, target["thread_ts"], fresh=False)
        except RoundNotOrderableError:
            if request.get("mix_action_key") is not None:
                raise
            order = _resolve_request(request, target["thread_ts"], fresh=True)
    except RoundNotOrderableError as exc:
        note = _ALREADY_DONE_NOTE if exc.done else f"⌛ 주문할 수 없는 신호입니다: {exc}"
        _update_view(view_id, _status_view(request, target, note))
        _expire(target, exc)
        return
    except Exception as exc:  # 모달은 이미 열렸다 — 실패 이유를 모달에 보이고 버튼은 다시 누를 수 있게 둔다.
        logger.exception("슬랙 주문 확인 모달 채우기 실패")
        _update_view(view_id, _status_view(request, target, f"❌ 주문 확인을 열 수 없습니다: {exc}"))
        return
    if request.get("mix_action_key") is not None:
        from utils.strategy_mix_orders import mix_order_terms, save_mix_confirmation

        save_mix_confirmation(request, target["thread_ts"], view_id)
        target = {**target, "confirmed": mix_order_terms(order), "confirmation_id": view_id}
    view = _swap_view(request, order, target) if _is_swap(request) else _order_view(request, order, target)
    _update_view(view_id, view)


def open_confirm_modal(payload: dict[str, Any]) -> dict[str, Any] | None:
    """버튼을 누른 사용자에게 확인 모달을 연다 — trigger_id 는 3초 안에 써야 한다.

    주문 버튼은 계산 없이 버튼에 담긴 주문 내용으로 「확인 중」 모달을 바로 열고, 호가·수량을 채울 일
    (`fill_confirm_modal` 에 넘길 {target, view_id})을 돌려준다. 취소 버튼은 계산할 것이 없어 바로 연다.
    """
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
    client = _slack_client()
    if action["action_id"] == OPEN_ACTION_ID:
        opened = client.views_open(
            trigger_id=payload["trigger_id"],
            view=_status_view(request, target, "⏳ 호가와 주문 가능 수량을 확인하는 중…"),
        )
        return {"target": target, "view_id": opened["view"]["id"]}
    if action["action_id"] == MODIFY_ACTION_ID:
        opened = client.views_open(
            trigger_id=payload["trigger_id"],
            view=_status_view(request, target, "⏳ 정정 호가와 잔여 수량을 확인하는 중…"),
        )
        return {"kind": "modify", "target": {**target, "order_id": value["order_id"]}, "view_id": opened["view"]["id"]}
    if action["action_id"] == CANCEL_ACTION_ID:
        client.views_open(
            trigger_id=payload["trigger_id"], view=_cancel_view(request, {**target, "order_id": value["order_id"]})
        )
        return None
    raise BrokerApiError(f"알 수 없는 슬랙 버튼입니다: {action['action_id']}")


def submitted_request(payload: dict[str, Any]) -> dict[str, Any]:
    """모달 제출 payload 에서 처리할 일(주문·정정·취소)과 대상 메시지를 꺼낸다."""
    view = payload["view"]
    kinds = {ORDER_CALLBACK_ID: "order", MODIFY_CALLBACK_ID: "modify", CANCEL_CALLBACK_ID: "cancel"}
    if view["callback_id"] not in kinds:
        raise BrokerApiError(f"알 수 없는 슬랙 모달입니다: {view['callback_id']}")
    return {**json.loads(view["private_metadata"]), "kind": kinds[view["callback_id"]]}


# --- 처리 -----------------------------------------------------------------------------------------------


def _final_text(order: dict[str, Any]) -> str:
    side = f"{_env_word(order['env'])} {_action_word(order)}"
    head = f"{_round_text(order['round'])}{order['strategy_label']} {order['quantity']:,}주 × {format_money(order['price'], order['country'])}"
    fill = f"체결 {order['filled_quantity']:,}주 · 체결가 {format_money(order['average_fill_price'], order['country'])}"
    if order["status"] == "filled":
        return f"✅ {side} 체결 완료: {head} ({fill})"
    if order["status"] == "canceled":
        return f"🚫 {side} 주문 취소됨: {head} ({fill})"
    if order["status"] == "expired":
        return f"⌛ {side} 주문 만료됨: {head} ({fill})"
    return f"❌ {side} 주문 거부됨: {head}" + (f" — {order['message']}" if order["message"] else "")


def _finish(target: dict[str, Any], order: dict[str, Any]) -> None:
    """끝난 주문의 결과를 스레드로 알리고 메시지에서 버튼을 걷어낸다 — 하나도 체결 안 된 취소만 다시 주문하기 버튼을 남긴다."""
    request = target["request"]
    text = _auto_tag(request) + _final_text(order)
    send_thread_message(text, thread_ts=target["thread_ts"])
    if order["pair_id"] is not None:
        _update_message(target, _pair_blocks(request, order["pair_id"], text))
        return
    if order["status"] == "canceled" and order["filled_quantity"] == 0:
        # 다시 주문해도 서버가 회차 보유 상태·중복·안전 장치를 처음 주문과 똑같이 검사한다.
        _update_message(target, [*_open_blocks(request, "다시 주문하기"), _note(text)])
        _release_reservation(target)
    else:
        _update_message(target, _note_blocks(request, text))


def _expire(target: dict[str, Any], exc: RoundNotOrderableError) -> None:
    """회차가 더는 주문 대상이 아니면 다시 눌러도 같은 결과이므로 버튼을 되살리지 않는다.

    다른 주문으로 이미 처리된 회차면 실패가 아니라 완료로 남기고(스레드 알림 없음), 아니면 주문 불가를 알린다.
    """
    request = target["request"]
    if request.get("mix_action_key") is not None:
        from utils.strategy_mix_orders import close_mix_signal

        close_mix_signal(request, target["thread_ts"])
    if exc.done:
        _update_message(target, _note_blocks(request, _ALREADY_DONE_NOTE))
        _settle_message(target, MESSAGE_DONE)
        return
    side = f"{_auto_tag(request)}{_env_word(request['env'])} {_action_word(request)}"
    send_thread_message(f"❌ {side} 주문 불가: {exc}", thread_ts=target["thread_ts"])
    _update_message(target, _note_blocks(request, "⌛ 보유 상태가 바뀌어 주문 불가"))
    _settle_message(target, MESSAGE_FAILED)


def _settle_message(target: dict[str, Any], outcome: str) -> None:
    """주문 없이 끝난 신호 스레드의 주문 메시지 결과를 남기고 신호 메시지 반응을 맞춘다(신호 스레드가 아니면 할 일 없음)."""
    thread = find_signal_thread(target["thread_ts"])
    if thread is None or all(message["ts"] != target["message_ts"] for message in thread["messages"]):
        return
    settle_signal_message(target["thread_ts"], target["message_ts"], outcome)
    _sync_thread(target["thread_ts"])


def _sync_thread(thread_ts: str) -> None:
    thread = find_signal_thread(thread_ts)
    if thread is not None:
        sync_signal_thread(thread)


def _release_reservation(target: dict[str, Any]) -> None:
    """주문이 나가지 못했거나 체결 없이 취소돼 버튼을 되살릴 때 — 회수를 막는 주문이 없으면 신호를 미사용으로 되돌려 장 마감 회수가 버튼을 걷게 한다."""
    from utils.strategy_trade_threads import release_signal_reservation

    release_signal_reservation(target["thread_ts"])


def _run_post_order_updates(target: dict[str, Any], thread_text: str, blocks: list[dict[str, Any]]) -> None:
    """주문 접수 직후 독립적인 슬랙 알림·버튼 전환·반응 동기화를 병렬로 처리한다."""

    def _safe(fn, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.exception("슬랙 주문 후속 업데이트 실패")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_safe, send_thread_message, thread_text, thread_ts=target["thread_ts"]),
            executor.submit(_safe, _update_message, target, blocks),
            executor.submit(_safe, _sync_thread, target["thread_ts"]),
        ]
        for future in futures:
            future.result()


def _place_swap(target: dict[str, Any], *, reraise: bool = False) -> dict[str, Any] | None:
    """매도 후 재매수 한 짝을 낸다 — 결과를 스레드로 알리고 메시지에 양쪽 취소 버튼을 단다."""
    request = target["request"]
    word = f"{_auto_tag(request)}{_env_word(request['env'])} 매도 후 재매수"
    _update_message(target, _note_blocks(request, "⏳ 주문 처리 중…"))
    try:
        result = place_swap(
            request["account_id"],
            request["pool"],
            request["strategy_id"],
            request["round"],
            _slack_of(target),
        )
    except RoundNotOrderableError as exc:
        _expire(target, exc)
        if reraise:
            raise BrokerApiError(str(exc)) from exc
        return None
    except Exception as exc:  # 슬랙 요청은 이미 응답했으므로 실패도 스레드로 알린다.
        logger.exception("슬랙 매도 후 재매수 처리 실패")
        send_thread_message(f"❌ {word} 주문 실패: {exc}", thread_ts=target["thread_ts"])
        _update_message(target, _open_blocks(request))
        _release_reservation(target)
        if reraise:
            raise
        return None
    sell, buy = result["sell"], result["buy"]
    text = f"✅ {word} 접수: {sell['strategy_label']} — 매도 {sell['quantity']:,}주 × {format_money(sell['price'], sell['country'])}"
    if buy is not None:
        text += f" · 매수 {buy['quantity']:,}주 × {format_money(buy['price'], buy['country'])}"
    else:
        text += f"\n⚠️ 매수 주문은 실패했습니다(매도만 걸림): {result['buy_error']}"
    request = _with_venue(request, sell["venue"])
    _run_post_order_updates({**target, "request": request}, text, _pair_blocks(request, sell["pair_id"], None))
    return sell


def _place(target: dict[str, Any], *, reraise: bool = False) -> dict[str, Any] | None:
    request = target["request"]
    if _is_swap(request):
        return _place_swap(target, reraise=reraise)
    side = f"{_auto_tag(request)}{_env_word(request['env'])} {_action_word(request)}"
    # 처리하는 동안 같은 메시지로 또 주문하지 못하게 버튼부터 걷어낸다.
    _update_message(target, _note_blocks(request, "⏳ 주문 처리 중…"))
    try:
        slack = _slack_of(target)
        if request.get("mix_action_key") is not None:
            from utils.strategy_mix_orders import place_mix_order

            order = place_mix_order(request, slack, target["confirmed"], target["confirmation_id"])
        else:
            order = place_order(
                request["account_id"],
                request["pool"],
                request["strategy_id"],
                request["side"],
                request["round"],
                slack,
                arbitrary_id=request.get("arbitrary_id"),
            )
    except OrderUncertainError as exc:
        text = "⚠️ 증권사 접수 여부를 확인 중입니다. 주문내역을 확인하세요."
        send_thread_message(text, thread_ts=target["thread_ts"])
        _update_message(target, _note_blocks(request, text))
        if reraise:
            raise BrokerApiError(str(exc)) from exc
        return None
    except RoundNotOrderableError as exc:
        _expire(target, exc)
        if reraise:
            raise BrokerApiError(str(exc)) from exc
        return None
    except Exception as exc:  # 슬랙 요청은 이미 응답했으므로 실패도 스레드로 알린다.
        logger.exception("슬랙 모의 주문 처리 실패")
        send_thread_message(f"❌ {side} 주문 실패: {exc}", thread_ts=target["thread_ts"])
        _update_message(target, _open_blocks(request))
        _release_reservation(target)
        if reraise:
            raise
        return None
    gap_pct = _signal_gap_pct(order)
    gap = f" · 신호 대비 {gap_pct:+.2f}%" if gap_pct is not None else ""
    request = _with_venue(request, order["venue"])
    _run_post_order_updates(
        {**target, "request": request},
        f"✅ {side} 주문 접수: {order['strategy_label']} {order['quantity']:,}주 × {format_money(order['price'], order['country'])} (지정가){gap}",
        _cancel_blocks(request, order["id"]),
    )
    return order


def _restore_order_buttons(target: dict[str, Any], order: dict[str, Any], note: str | None = None) -> None:
    request = target["request"]
    if order["pair_id"] is not None:
        swap = {**request, "side": "sell", "rebuy": True}
        _update_message({**target, "request": swap}, _pair_blocks(swap, order["pair_id"], note))
    elif order["status"] in FINISHED_STATUSES:
        _update_message(target, _note_blocks(request, note or "⏳ 결과 확인 중…"))
    else:
        blocks = _cancel_blocks(request, order["id"])
        _update_message(target, [*blocks, _note(note)] if note else blocks)


def _modify(target: dict[str, Any], *, reraise: bool = False) -> dict[str, Any] | None:
    request = target["request"]
    side = f"{_env_word(request['env'])} {_SIDE_LABELS[request['side']]}"
    _update_message(target, _note_blocks(request, "⏳ 호가 정정 처리 중…"))
    try:
        modified = modify_record(target["order_id"])
    except Exception as exc:
        logger.exception("슬랙 주문 호가 정정 실패")
        send_thread_message(f"❌ {side} 호가 정정 실패: {exc}", thread_ts=target["thread_ts"])
        _restore_order_buttons(target, refreshed_order(request["account_id"], target["order_id"]))
        if reraise:
            raise
        return None
    text = f"✏️ {side} 호가 정정 요청: {format_money(modified['price'], modified['country'])}"
    send_thread_message(text, thread_ts=target["thread_ts"])
    _restore_order_buttons(target, modified, text)
    return modified


def _cancel(target: dict[str, Any], *, reraise: bool = False) -> dict[str, Any] | None:
    request = target["request"]
    side = f"{_env_word(request['env'])} {_SIDE_LABELS[request['side']]}"
    _update_message(target, _note_blocks(request, "⏳ 취소 처리 중…"))
    try:
        cancelled = cancel_record(target["order_id"])
    except Exception as exc:
        logger.exception("슬랙 모의 주문 취소 실패")
        send_thread_message(f"❌ {side} 주문 취소 실패: {exc}", thread_ts=target["thread_ts"])
        # 그 사이 주문이 끝났을 수 있다 — 끝났으면 결과 알림은 배치가 하고, 아니면 취소 버튼을 되살린다.
        _restore_order_buttons(target, refreshed_order(request["account_id"], target["order_id"]))
        if reraise:
            raise
        return None
    send_thread_message(f"🚫 {side} 주문 취소를 요청했습니다.", thread_ts=target["thread_ts"])
    # 취소 결과(체결 수량 포함)로 메시지를 마무리하는 일은 배치(`notify_finished_orders`)가 한다.
    return cancelled


def _submit_order(target: dict[str, Any]) -> None:
    """확인된 주문 한 건을 신호 선점 뒤 낸다 — 버튼 확인과 자동 주문이 같이 쓴다. 호출하는 쪽이 `signal_lock` 을 잡고 있어야 한다."""
    from utils.strategy_trade_threads import finish_order_reservation, reserve_signal_for_order

    if target["request"].get("mix_action_key") is not None:
        from utils.strategy_mix_orders import mix_submission_open

        if not mix_submission_open(target["request"], target["thread_ts"], target["confirmation_id"]):
            return
    if not reserve_signal_for_order(target["thread_ts"], target["message_ts"]):
        return
    try:
        _place(target)
    finally:
        finish_order_reservation(target["thread_ts"])


def run_submission(target: dict[str, Any]) -> None:
    """확인된 주문·정정·취소를 처리한다 — 응답 뒤 백그라운드에서 돈다."""
    try:
        with signal_lock():
            if target["kind"] == "order":
                _submit_order(target)
            elif target["kind"] == "modify":
                _modify(target)
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
        # 자동 주문 표시가 생기기 전 주문 기록에는 키가 없다 — 그것들은 모두 수동 주문이다.
        "auto": record["slack"].get("auto") is True,
        "venue": record["venue"],
        **({"mix_action_key": record["mix_action_key"]} if "mix_action_key" in record else {}),
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
        text = f"⏰ {side} 주문이 아직 체결되지 않았습니다(미체결 {remaining:,}주). {close_warning_text(record)} 마감 전에 취소하지 않으면 오늘 주문은 만료됩니다."
        send_thread_message(text, thread_ts=target["thread_ts"])
        if order["pair_id"] is not None:
            _update_message(target, _pair_blocks(request, order["pair_id"], text))
        else:
            _update_message(target, [*_cancel_blocks(request, order["id"]), _note(text)])
        mark_slack_close_warned(record["_id"])
    return len(records)


def _react(method: str, channel: str, ts: str, name: str) -> None:
    """메시지 반응을 달거나 뗀다 — 반응은 부가 표시라 실패해도 로그만 남기고 흐름을 막지 않는다(권한: reactions:write)."""
    client = _slack_client()
    try:
        getattr(client, method)(channel=channel, timestamp=ts, name=name)
    except SlackApiError as exc:
        error = exc.response.get("error")
        if error not in _REACTION_NOOP_ERRORS:
            logger.error("슬랙 반응 %s(%s) 실패: %s", method, name, error)


def sync_signal_thread(thread: dict[str, Any]) -> str | None:
    """신호 메시지 반응을 스레드 상태 하나(⏳·✅·❌·없음)로 맞춘다 — 다른 반응은 뗀다. 완료한 스레드는 추적을 끝낸다.

    반응을 다는 곳은 이 함수 하나다. 상태는 주문 메시지별 주문 결과와 남긴 결과로 정한다(`signal_thread_state`).
    """
    state = signal_thread_state(signal_message_states(thread))
    channel, thread_ts = thread["channel"], thread["thread_ts"]
    if state != thread["reaction"]:
        to_remove = [name for key, name in REACTIONS.items() if key != state]
        if to_remove:
            with ThreadPoolExecutor(max_workers=len(to_remove)) as executor:
                for future in [
                    executor.submit(_react, "reactions_remove", channel, thread_ts, name) for name in to_remove
                ]:
                    future.result()
        if state is not None:
            _react("reactions_add", channel, thread_ts, REACTIONS[state])
        set_signal_thread_reaction(thread_ts, state)
    if state == MESSAGE_DONE:
        from utils.strategy_trade_threads import complete_signal_thread

        complete_signal_thread(thread_ts)
        forget_signal_thread(thread_ts)
    return state


def sync_signal_threads() -> int:
    """추적 중인 신호 스레드의 반응을 모두 맞춘다(체결·취소 결과가 반영된 뒤) — 완료한 스레드 수."""
    return sum(sync_signal_thread(thread) == MESSAGE_DONE for thread in signal_threads())


def close_session_threads(strategies: list[dict[str, Any]]) -> int:
    """세션이 닫힌 전략의 신호 스레드에서 남은 주문 버튼을 걷는다 — 정리한 메시지 수.

    주문 없이 열린 버튼은 이미 처리된 회차면 완료, 아니면 장 마감 실패로 남기고, 체결 없이 취소돼 다시 주문하기
    버튼이 남은 메시지도 장 마감 문구로 바꾼다. 진행 중인 주문이 없는 스레드는 반응을 맞춘 뒤 추적을 끝낸다.
    주문이 하나도 없는 미사용 신호는 회수(`withdraw_signal`)가 먼저 정리한다.
    """
    from utils.strategy_trade_orders import reorderable_signal_messages

    by_id = {strategy["strategy_id"]: strategy for strategy in strategies if "error" not in strategy}
    closed = 0
    for thread in signal_threads():
        strategy = by_id.get(thread["messages"][0]["request"]["strategy_id"])
        if strategy is None:
            continue
        thread_ts = thread["thread_ts"]
        from utils.strategy_trade_threads import signal_is_manual

        if signal_is_manual(thread_ts):
            continue
        states = signal_message_states(thread)
        settled = {item["ts"] for item in thread["settled"]}
        reorderable = reorderable_signal_messages(thread)
        for message in thread["messages"]:
            ts, request = message["ts"], message["request"]
            target = {"request": request, "channel": thread["channel"], "message_ts": ts, "thread_ts": thread_ts}
            if states[ts] == MESSAGE_OPEN:
                done = round_done(strategy, request)
                _update_message(target, _note_blocks(request, _ALREADY_DONE_NOTE if done else _SESSION_CLOSED_NOTE))
                settle_signal_message(thread_ts, ts, MESSAGE_DONE if done else MESSAGE_FAILED)
                closed += 1
            elif ts in reorderable and ts not in settled:
                _update_message(target, _note_blocks(request, _SESSION_CLOSED_NOTE))
                settle_signal_message(thread_ts, ts, MESSAGE_FAILED)
                closed += 1
        refreshed = find_signal_thread(thread_ts)
        if refreshed is not None and sync_signal_thread(refreshed) not in (MESSAGE_PENDING, MESSAGE_PLACED):
            forget_signal_thread(thread_ts)
    return closed


def send_signal_order_messages(strategy_view: dict[str, Any], triggers: list[dict[str, Any]], signal_ts: str) -> int:
    """신호가 난 회차마다 주문 메시지(버튼)를 신호 메시지의 스레드로 보내고 그 스레드를 추적 대상으로 기록한다 — 보낸 건수.

    이미 주문이 진행 중인 회차는 건너뛴다. 주문은 슬랙의 버튼과 모달 확인을 거쳐야 나간다 — 단 「자동」을 켠
    전략의 자동 신호(`is_auto_trigger`)는 메시지를 보낸 뒤 버튼 확인과 같은 경로(`_submit_order`)로 바로 주문한다.
    호출하는 쪽이 `signal_lock` 을 잡고 있어야 한다.
    """
    account_id = strategy_view["account_id"]
    pending = [
        trigger
        for trigger in triggers
        if find_duplicate_order(account_id, trigger["strategy_id"], trigger["action"]) is None
        # 매도 후 재매수는 같은 회차의 매수 주문도 걸리므로 그쪽 중복도 본다.
        and not (trigger["rebuy"] and find_duplicate_order(account_id, trigger["strategy_id"], "buy") is not None)
    ]
    messages: list[dict[str, Any]] = []
    try:
        for trigger in pending:
            if trigger.get("origin") == "manual":
                message = _send_order_request(
                    account_id,
                    strategy_view["pool"],
                    trigger["strategy_id"],
                    trigger["action"],
                    trigger["round"],
                    signal_ts,
                    False,
                    None,
                    {
                        "strategy_label": strategy_view["label"],
                        "ticker": trigger["ticker"],
                        "quantity": trigger["quantity"],
                    },
                    auto=False,
                )
                messages.append(message)
                continue
            message = send_order_message(
                account_id,
                strategy_view["pool"],
                trigger["strategy_id"],
                trigger["action"],
                trigger["round"],
                signal_ts,
                rebuy=trigger["rebuy"],
                auto=is_auto_trigger(strategy_view, trigger),
                arbitrary_id=trigger.get("arbitrary_id"),
            )
            messages.append(message)
    finally:
        # 중간에 실패해도 이미 보낸 메시지는 스레드로 추적한다.
        if messages:
            register_signal_thread(get_slack_channel(), signal_ts, messages)
    channel = get_slack_channel()
    for message in messages:
        if message["request"]["auto"]:
            # 사람이 버튼을 누르고 확인한 것과 같은 대상 — 주문 실패는 `_place` 가 스레드로 알리고 버튼을 되살린다.
            try:
                _submit_order(
                    {
                        "request": message["request"],
                        "channel": channel,
                        "message_ts": message["ts"],
                        "thread_ts": signal_ts,
                        "kind": "order",
                    }
                )
            except Exception as exc:  # 한 건이 멈춰도 나머지 자동 주문은 진행한다.
                logger.exception("자동 주문 처리 실패")
                send_thread_message(f"⚠️ {AUTO_ORDER_TAG} 처리 중 오류로 멈춥니다: {exc}", thread_ts=signal_ts)
    return len(messages)


# --- 웹 통합 모달 공용 조회·실행 (합성 전략·전략 사고팔기 공용) ------------------------------------------


def order_summary(order: dict[str, Any]) -> dict[str, Any]:
    filled = int(order.get("filled_quantity") or 0)
    quantity = int(order["quantity"])
    return {
        "id": order.get("id") or str(order["_id"]),
        "status": order["status"],
        "price": order["price"],
        "quantity": quantity,
        "filled_quantity": filled,
        "remaining_quantity": max(quantity - filled, 0),
        "country": order["country"],
    }


def _match_thread_message(
    thread: dict[str, Any],
    *,
    side: str | None = None,
    round_no: int | None = None,
    ticker: str | None = None,
) -> dict[str, Any] | None:
    for item in thread.get("messages") or []:
        req = item.get("request") or {}
        if side is not None and req.get("side") != side:
            continue
        if round_no is not None and req.get("round") != round_no:
            continue
        if ticker is not None and req.get("ticker") != ticker:
            continue
        return item
    return (thread.get("messages") or [None])[0]


def describe_slack_thread_action(
    account_id: str,
    thread_ts: str,
    *,
    side: str | None = None,
    round_no: int | None = None,
    ticker: str | None = None,
    allow_unregistered_withdraw: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """슬랙 스레드 기반 웹 모달의 현재 상태(주문 전·미체결·처리 완료)를 공용 판정한다.

    반환: ``(state_dict, request_or_none, resolved_order_or_none)``.
    """
    order_query: dict[str, Any] = {"slack.thread_ts": thread_ts, "account_id": account_id}
    if ticker is not None:
        order_query["ticker"] = ticker
    if side is not None:
        order_query["side"] = side
    if round_no is not None:
        order_query["round"] = round_no
    latest = _orders().find_one(order_query, sort=[("_id", -1)])
    if latest and latest["status"] in ACTIVE_STATUSES:
        refreshed = refreshed_order(account_id, str(latest["_id"]))
        if refreshed["status"] in ACTIVE_STATUSES:
            preview = None
            modify_error = None
            try:
                preview = preview_modify_record(refreshed["id"])
            except BrokerApiError as exc:
                modify_error = str(exc)
            return (
                {
                    "phase": "active_order",
                    "status_label": "미체결 진행 중",
                    "thread_ts": thread_ts,
                    "order": order_summary(refreshed),
                    "modify_preview": {
                        "target_price": preview["target_price"],
                        "quote_label": preview["quote_label"],
                        "remaining_quantity": preview["remaining_quantity"],
                    }
                    if preview
                    else None,
                    "modify_error": modify_error,
                    "withdraw_key": None,
                },
                None,
                None,
            )
        latest = refreshed
    elif latest:
        latest = serialize_order(latest)

    if latest and (int(latest.get("filled_quantity") or 0) > 0 or latest["status"] in ("filled", "expired")):
        return (
            {
                "phase": "completed",
                "status_label": "처리 완료",
                "reason": f"이미 주문({latest['status']})이 처리되었습니다.",
                "order": order_summary(latest),
                "withdraw_key": None,
            },
            None,
            None,
        )

    withdraw_key = thread_ts if can_withdraw_signal(thread_ts, allow_unregistered=allow_unregistered_withdraw) else None
    thread = find_signal_thread(thread_ts)
    message = _match_thread_message(thread, side=side, round_no=round_no, ticker=ticker) if thread else None
    if not message:
        return (
            {
                "phase": "completed",
                "status_label": "신호 정보 없음",
                "reason": "슬랙 스레드 정보를 찾을 수 없습니다.",
                "withdraw_key": withdraw_key,
            },
            None,
            None,
        )

    request = message["request"]
    try:
        try:
            order = _resolve_request(request, thread_ts, fresh=False)
        except RoundNotOrderableError:
            if request.get("mix_action_key") is not None:
                raise
            order = _resolve_request(request, thread_ts, fresh=True)
    except RoundNotOrderableError as exc:
        return (
            {
                "phase": "completed",
                "status_label": "주문 불가",
                "reason": str(exc),
                "withdraw_key": withdraw_key,
            },
            request,
            None,
        )
    except BrokerApiError as exc:
        return (
            {
                "phase": "ready_to_order",
                "status_label": "주문 대기",
                "thread_ts": thread_ts,
                "order_error": str(exc),
                "withdraw_key": withdraw_key,
            },
            request,
            None,
        )

    if _is_swap(request):
        price = order["sell_price"]
        quantity = order["sell_quantity"]
        quote_label = "매도 1호가 · 재매수 1호가"
        venue = order["venue"]
        signal_gap_pct = (
            (order["sell_price"] / order["signal_price"] - 1.0) * 100.0 if order.get("signal_price") else None
        )
    else:
        price = order["price"]
        quantity = order["quantity"]
        quote_label = order.get("quote_label") or ("매도 1호가" if request["side"] == "buy" else "매수 1호가")
        venue = order["venue"]
        signal_gap_pct = _signal_gap_pct(order)

    is_trade = request.get("mix_action_key") is None
    return (
        {
            "phase": "ready_to_order",
            "status_label": "주문 대기",
            "thread_ts": thread_ts,
            "order_preview": {
                "quantity": quantity,
                "price": price,
                "amount": price * quantity,
                "quote_label": quote_label,
                "venue": venue,
                "country": order["country"],
                "signal_gap_pct": signal_gap_pct if is_trade else None,
                "stock_profit_pct": order.get("stock_profit_pct") if is_trade else None,
                "average_price": order.get("average_price") if is_trade else None,
                "round_profit_pct": order.get("round_profit_pct") if is_trade else None,
                "round_fill_price": order.get("round_fill_price") if is_trade else None,
            },
            "withdraw_key": withdraw_key,
        },
        request,
        order,
    )


def execute_slack_thread_action(
    kind: str,
    account_id: str,
    thread_ts: str,
    *,
    order_id: str | None = None,
    ticker: str | None = None,
    side: str | None = None,
    round_no: int | None = None,
    confirmed: dict[str, Any] | None = None,
    confirmation_id: str | None = None,
    before_order: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> None:
    """웹 통합 모달의 주문·호가정정·취소를 슬랙과 동일한 락·스레드 동기화 경로로 실행한다."""
    with signal_lock():
        if kind in ("modify", "cancel"):
            if not order_id:
                raise BrokerApiError("대상 주문 ID가 없습니다.")
            query: dict[str, Any] = {"_id": ObjectId(order_id), "account_id": account_id}
            if ticker is not None:
                query["ticker"] = ticker
            record = _orders().find_one(query)
            if record is None:
                raise BrokerApiError("주문 기록을 찾을 수 없습니다.")
            refreshed = refreshed_order(account_id, order_id)
            if refreshed["status"] not in ACTIVE_STATUSES:
                raise BrokerApiError("이미 체결되었거나 취소된 주문입니다.")
            target = {**_target_of(record), "order_id": order_id}
            if kind == "modify":
                _modify(target, reraise=True)
            else:
                _cancel(target, reraise=True)
            return

        if kind != "order":
            raise BrokerApiError(f"지원하지 않는 요청 종류입니다: {kind}")
        thread = find_signal_thread(thread_ts)
        message = _match_thread_message(thread, side=side, round_no=round_no, ticker=ticker) if thread else None
        if not thread or not message:
            raise BrokerApiError("이미 회수되었거나 스레드 정보를 찾을 수 없습니다.")
        request = message["request"]
        if before_order is not None:
            before_order(thread, message)
        if not reserve_signal_for_order(thread_ts, message["ts"]):
            raise BrokerApiError("이미 다른 주문이 진행 중입니다. 잠시 후 다시 확인하세요.")
        target: dict[str, Any] = {
            "request": request,
            "channel": thread["channel"],
            "message_ts": message["ts"],
            "thread_ts": thread_ts,
        }
        if confirmed is not None:
            target["confirmed"] = confirmed
        if confirmation_id is not None:
            target["confirmation_id"] = confirmation_id
        try:
            _place(target, reraise=True)
        finally:
            finish_order_reservation(thread_ts)
