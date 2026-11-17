"""합성 오늘의 액션을 기존 Slack 주문 흐름에 연결한다."""

from __future__ import annotations

from typing import Any

from services.broker_api_service import BrokerApiError, OrderUncertainError, fetch_broker_balance
from services.broker_trade_service import account_env, list_day_orders, resolve_us_exchange
from utils.db_manager import get_db_connection
from utils.market_session import REGULAR, market_session
from utils.settings_loader import get_account_settings, get_slack_channel
from utils.strategy_mix_notify import format_action_message
from utils.strategy_mix_positions import mix_positions
from utils.strategy_trade_market import broker_order_date
from utils.strategy_trade_orders import (
    ACTIVE_STATUSES,
    RoundNotOrderableError,
    _now,
    _orders,
    _require_order_allowed,
    register_signal_thread,
    resolve_order_quote,
    submit_order_record,
)
from utils.strategy_trade_signal_lock import signal_lock
from utils.strategy_trade_threads import can_withdraw_signal, send_thread_message, withdraw_signal
from utils.ticker_resolver import resolve_ticker_meta

COLLECTION = "strategy_mix_order_signals"


class MixOrderChangedError(BrokerApiError):
    """Slack에서 확인한 주문 조건이 바뀌어 다시 확인해야 한다."""


def _signals():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db[COLLECTION]


def _signal_id(account_id: str, ticker: str) -> dict[str, str]:
    return {"account_id": account_id, "ticker": ticker}


def _today_action(account_id: str, action_key: str) -> tuple[dict, dict, dict]:
    positions = mix_positions(account_id)
    if positions["account_id"] != account_id:
        raise BrokerApiError("선택한 합성 계좌를 확인할 수 없습니다.")
    for group in positions["actions"]["groups"]:
        if group["section"] != "today":
            continue
        for item in group["items"]:
            if item["key"] == action_key and item["quantity"] > 0:
                return positions, group, item
    raise RoundNotOrderableError("이 종목은 더 이상 오늘의 주문 대상이 아닙니다.", done=False)


def _require_live_account(account_id: str) -> str:
    env = account_env(account_id)
    if env != "live":
        raise BrokerApiError("합성 주문은 운영 계좌에서만 사용할 수 있습니다.")
    _require_order_allowed(account_id, env, 0, 1)
    return env


def _require_no_open_order(account_id: str, ticker: str, country: str) -> None:
    if _orders().find_one({"account_id": account_id, "ticker": ticker, "status": {"$in": list(ACTIVE_STATUSES)}}):
        raise BrokerApiError("이 종목에 이미 진행 중이거나 접수 확인이 필요한 API 주문이 있습니다.")
    broker_orders = list_day_orders(account_id, broker_order_date(country, _now()))
    if any(row["ticker"] == ticker and row["status"] in ACTIVE_STATUSES for row in broker_orders):
        raise BrokerApiError("이 종목에 증권사 미체결 주문이 있습니다.")


def _checked_balance(positions: dict, ticker: str) -> dict:
    balance = fetch_broker_balance(positions["account_id"])
    broker_quantity = sum(row["quantity"] for row in balance["holdings"] if row["ticker"] == ticker)
    account_holding = positions["account"]["holdings"].get(ticker)
    if broker_quantity != (account_holding["quantity"] if account_holding else 0):
        raise BrokerApiError("증권사 보유와 합성 보유가 다릅니다. 잔고 동기화 뒤 다시 확인하세요.")
    return balance


def _used_signal(record: dict) -> bool:
    if not record.get("thread_ts"):
        return False
    # 체결분이 있거나 당일 주문이 끝났다면 기존 버튼으로 다시 주문하지 않는다.
    return (
        _orders().find_one(
            {
                "slack.thread_ts": record["thread_ts"],
                "$or": [{"filled_quantity": {"$gt": 0}}, {"status": {"$in": ["filled", "expired"]}}],
                "status": {"$nin": list(ACTIVE_STATUSES)},
            }
        )
        is not None
    )


def close_mix_signal(request: dict, thread_ts: str) -> None:
    _signals().update_one(
        {"_id": _signal_id(request["account_id"], request["ticker"]), "thread_ts": thread_ts},
        {"$set": {"state": "closed"}},
    )


def _current_signal(request: dict, thread_ts: str) -> None:
    record = _signals().find_one({"_id": _signal_id(request["account_id"], request["ticker"])})
    if (
        not record
        or record.get("thread_ts") != thread_ts
        or record["action_key"] != request["mix_action_key"]
        or record["side"] != request["side"]
        or record["state"] != "sent"
        or _used_signal(record)
    ):
        raise RoundNotOrderableError("이미 종료되었거나 다른 신호로 대체된 주문입니다.", done=False)


def _signal_status(record: dict) -> dict[str, str | None]:
    thread_ts = record.get("thread_ts")
    return {
        "state": record["state"],
        "withdraw_key": thread_ts
        if record["state"] == "sent" and thread_ts and can_withdraw_signal(thread_ts, allow_unregistered=True)
        else None,
    }


@signal_lock()
def withdraw_mix_order_signal(account_id: str, ticker: str, thread_ts: str) -> None:
    """같은 종목의 새 발송과 경합해도 확인창이 가리키는 메시지만 회수한다."""
    collection = _signals()
    query = {"_id": _signal_id(account_id, ticker), "thread_ts": thread_ts, "state": "sent"}
    record = collection.find_one(query)
    if record is None:
        raise BrokerApiError("이미 회수됐거나 다른 신호로 바뀌었습니다.")
    name = record.get("name")
    if not name:
        # 기존 발송본은 스레드의 주문 등록에서 종목명을 읽는다.
        from utils.strategy_trade_orders import find_signal_thread

        thread = find_signal_thread(thread_ts)
        name = next((item["request"]["label"] for item in thread["messages"]), ticker) if thread else ticker
    entry = {"strategy_id": "mix", "name": name, "round": None, "action": record["side"], "origin": "manual"}
    if not withdraw_signal(thread_ts, "manual", None, entry=entry):
        raise BrokerApiError("이 신호로 낸 주문이 있거나 이미 사용되어 회수할 수 없습니다.")
    collection.delete_one(query)


def mix_order_signals(account_id: str, groups: list[dict]) -> dict[str, dict[str, str | None]]:
    """오늘 화면에 남아 있는 신호의 발송 상태를 조회한다."""
    actions = {item["key"]: item["side"] for group in groups if group["section"] == "today" for item in group["items"]}
    if not actions:
        return {}
    result = {}
    for record in _signals().find({"account_id": account_id, "action_key": {"$in": list(actions)}}):
        key = record["action_key"]
        if record["side"] == actions[key] and record["state"] != "closed" and not _used_signal(record):
            result[key] = _signal_status(record)
    return result


@signal_lock()
def send_mix_order_signal(account_id: str, action_key: str, side: str, quantity: int) -> dict:
    """웹에서 다시 확인한 오늘의 지시 한 건을 Slack 주문 스레드로 보낸다."""
    from utils.slack_mock_orders import send_order_request

    positions, group, item = _today_action(account_id, action_key)
    if item["side"] != side or item["quantity"] != quantity:
        raise BrokerApiError("오늘의 액션이 바뀌었습니다. 새로 불러온 뒤 확인하세요.")
    env = _require_live_account(account_id)
    ticker, country = item["ticker"], positions["country"]
    key = _signal_id(account_id, ticker)
    collection = _signals()
    current = collection.find_one({"_id": key})
    if (
        current
        and current["action_key"] == action_key
        and current["side"] == side
        and current["state"] != "closed"
        and not _used_signal(current)
    ):
        if current["state"] == "sent":
            return {"already_sent": True, "signal": _signal_status(current)}
        raise BrokerApiError("이 신호의 발송 결과를 확인해야 합니다. Slack을 확인하세요.")
    _require_no_open_order(account_id, ticker, country)
    _checked_balance(positions, ticker)
    request = {
        "env": env,
        "round": None,
        "account_id": account_id,
        "pool": "mix",
        "strategy_id": "mix",
        "side": side,
        "label": next(row["name"] for row in positions["holdings"] if row["ticker"] == ticker),
        "ticker": ticker,
        "quantity": quantity,
        "rebuy": False,
        "arbitrary_id": None,
        "mix_action_key": action_key,
    }
    account_name = get_account_settings(account_id).get("name") or account_id
    collection.replace_one(
        {"_id": key},
        {
            "_id": key,
            "account_id": account_id,
            "action_key": action_key,
            "side": side,
            "state": "sending",
            "name": request["label"],
        },
        upsert=True,
    )
    thread_ts = send_thread_message(format_action_message(account_name, [{**group, "items": [item]}]))
    if not thread_ts:
        raise BrokerApiError("Slack 발송 결과를 확인할 수 없습니다. Slack을 확인하세요.")
    collection.update_one({"_id": key}, {"$set": {"thread_ts": thread_ts}})
    message = send_order_request(request, thread_ts)
    register_signal_thread(get_slack_channel(), thread_ts, [message])
    collection.update_one({"_id": key}, {"$set": {"state": "sent"}})
    return {"already_sent": False, "signal": _signal_status({"state": "sent", "thread_ts": thread_ts})}


def resolve_mix_order(request: dict[str, Any], thread_ts: str) -> dict:
    """합성의 최신 지시와 증권사 잔고·호가로 확인창에 표시할 주문을 계산한다."""
    _current_signal(request, thread_ts)
    account_id, side, ticker = request["account_id"], request["side"], request["ticker"]
    positions, _, item = _today_action(account_id, request["mix_action_key"])
    if item["side"] != side or item["ticker"] != ticker:
        raise RoundNotOrderableError("오늘의 액션 방향이 달라졌습니다.", done=False)
    env = _require_live_account(account_id)
    country = positions["country"]
    if market_session(country, now=_now())["session"] != REGULAR:
        raise BrokerApiError("합성 주문은 해당 시장 정규장에만 가능합니다.")
    _require_no_open_order(account_id, ticker, country)
    balance = _checked_balance(positions, ticker)
    meta = resolve_ticker_meta(f"{country.upper()}:{ticker}")
    instrument = {
        "account_id": account_id,
        "ticker": ticker,
        "country": country,
        "label": meta["name"],
        "instrument_kind": "etf" if meta["is_etf"] else "stock",
        "exchange": resolve_us_exchange(account_id, ticker) if country == "us" else None,
    }
    venue, quote_label, price = resolve_order_quote(instrument, side)
    quantity = item["quantity"]
    amount = price * quantity
    if side == "buy" and amount > balance["cash"]:
        raise BrokerApiError("증권사 조회 예수금으로 이 매수 수량을 주문할 수 없습니다.")
    _require_order_allowed(account_id, env, amount if side == "buy" else -amount, 1)
    return {
        "env": env,
        "account_id": account_id,
        "pool": "mix",
        "strategy_id": "mix",
        "round": None,
        "mix_action_key": request["mix_action_key"],
        "ticker": ticker,
        "instrument_kind": instrument["instrument_kind"],
        "strategy_label": meta["name"],
        "country": country,
        "quantity": quantity,
        "price": price,
        "venue": venue,
        "quote_label": quote_label,
        "signal_price": None,
    }


def save_mix_confirmation(request: dict, thread_ts: str, confirmation_id: str) -> None:
    """마지막으로 열린 확인창 한 건을 기록한다."""
    _signals().update_one(
        {"_id": _signal_id(request["account_id"], request["ticker"]), "thread_ts": thread_ts, "state": "sent"},
        {"$set": {"confirmation_id": confirmation_id}},
    )


def mix_submission_open(request: dict, thread_ts: str, confirmation_id: str) -> bool:
    """소비한 확인창 제출은 기존 주문 메시지 상태를 건드리기 전에 거른다."""
    record = _signals().find_one(
        {
            "_id": _signal_id(request["account_id"], request["ticker"]),
            "thread_ts": thread_ts,
            "state": "sent",
            "confirmation_id": confirmation_id,
        }
    )
    return bool(
        record
        and not _used_signal(record)
        and not _orders().find_one(
            {
                "slack.thread_ts": thread_ts,
                "status": {"$in": list(ACTIVE_STATUSES)},
            }
        )
    )


def mix_order_terms(order: dict) -> dict:
    return {key: order[key] for key in ("quantity", "price", "venue")}


def place_mix_order(request: dict, slack: dict, confirmed: dict, confirmation_id: str) -> dict:
    order = resolve_mix_order(request, slack["thread_ts"])
    if confirmed != mix_order_terms(order):
        raise MixOrderChangedError("수량 또는 호가가 바뀌었습니다. 주문 버튼을 다시 눌러 확인하세요.")
    claimed = _signals().update_one(
        {
            "_id": _signal_id(request["account_id"], request["ticker"]),
            "thread_ts": slack["thread_ts"],
            "state": "sent",
            "confirmation_id": confirmation_id,
        },
        {"$unset": {"confirmation_id": ""}},
    )
    if not claimed.matched_count:
        raise MixOrderChangedError("이미 사용했거나 이전에 연 확인창입니다. 주문 버튼을 다시 눌러 확인하세요.")
    base = {key: value for key, value in order.items() if key not in ("price", "signal_price", "quote_label")}
    try:
        placed = submit_order_record({**base, "slack": slack}, request["side"], order["price"])
    except BrokerApiError:
        raise
    except Exception as exc:
        raise OrderUncertainError("합성 주문의 접수 여부를 확인하지 못했습니다.") from exc
    return {**placed, "signal_price": order["signal_price"]}
