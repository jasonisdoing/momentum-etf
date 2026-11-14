"""사고팔기 주문 기록 — 우리가 증권사에 보낸 주문과 그 상태(API 주문).

시험 주문은 회차 사다리와 무관하다. 전략의 종목을 **지금 상대 1호가를 지정가로** 사고판다(매수는 매도 1호가, 매도는
매수 1호가 — 마지막 체결가로 내면 체결이 안 되는 일이 잦다) — 거래 경로만 시험한다.
기록은 **무엇을 보냈는지**(전략·방향·가격·수량)를 남기고, 상태(대기중·부분체결·완료·취소·거부)는
**증권사 체결 조회가 진실**이라 조회할 때마다 거기서 갱신한다. 우리가 상태를 추측하지 않는다.

- 가격·수량은 클라이언트가 보내지 않는다. 서버가 증권사 1호가와 전략 운용 현황(`strategy_trade_service`)의
  매수는 회차 금액과 주문가로 수량을 계산하고 매도는 해당 회차의 남은 수량을 읽는다.
- 증권사는 계좌 설정의 연동으로 정한다(`services.broker_trade_service`, 나무·메리츠). 기록은 계좌 ID 로 묶는다.
- 주문번호 대응: 주문 응답의 주문번호를 그날 주문내역의 주문번호와 같은 번호로 본다(나무는 시장주문번호·통합주문번호가
  같은 번호 공간). 조회 결과에서 못 찾으면 상태를 바꾸지 않고 그 사실을 메시지로 남긴다 — 속성으로 짐작해 맞추지 않는다.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from bson import ObjectId

from core.strategy.trade.rules import is_valid_quote
from services.broker_api_service import BrokerApiError, OrderUncertainError
from services.broker_trade_service import (
    account_country,
    account_env,
    cancel_order,
    fetch_best_quote,
    list_day_orders,
    place_limit_order,
)
from utils.account_settings_store import ACCOUNT_ENV_MOCK, load_account_docs
from utils.db_manager import get_db_connection
from utils.strategy_trade_config import round_buy_quantity
from utils.strategy_trade_ledger import REGULAR_SELL, is_buy
from utils.strategy_trade_market import (
    broker_order_date,
    close_text,
    closing_windows,
    format_money,
    is_order_open,
    krw_rate,
    order_venue,
    session_trade_date,
)
from utils.strategy_trade_service import load_views_where
from utils.strategy_trade_threads import SIGNAL_RETENTION_DAYS

KST = ZoneInfo("Asia/Seoul")
COLLECTION = "strategy_trade_orders"
# 신호 스레드 기록은 사고팔기 설정 문서의 `signal_threads` 목록에 둔다 — 완료하면 지우고, 날짜가 바뀌어도 이어 추적한다.
_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"
#: 아직 끝나지 않은 상태 — 이 동안 체결 조회로 상태를 갱신한다.
ACTIVE_STATUSES = ("submitting", "pending", "partial")
#: 끝난 상태 — 만료는 접수일이 지났는데도 증권사가 미체결로 답하는 당일 주문이다.
FINISHED_STATUSES = ("filled", "canceled", "rejected", "expired")
#: 전송 결과를 모른 채 이 시간이 지난 기록은 거부로 정리한다.
_SUBMIT_TIMEOUT_SECONDS = 120


class RoundNotOrderableError(BrokerApiError):
    """주문하려던 회차가 이미 주문 대상이 아니다(보유 상태가 바뀜) — 다시 눌러도 같은 결과라 슬랙 버튼을 되살리지 않는다.

    ``done`` 은 그 회차가 이미 처리됐는지(다른 주문으로 체결됨)다 — 참이면 실패가 아니라 완료로 다룬다.
    """

    def __init__(self, message: str, *, done: bool) -> None:
        super().__init__(message)
        self.done = done


_index_ready = False


def _orders():
    global _index_ready
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    collection = db[COLLECTION]
    if not _index_ready:
        collection.create_index([("account_id", 1), ("created_at", -1)])
        _index_ready = True
    return collection


def _now() -> datetime:
    return datetime.now(KST)


def _require_round_state(strategy: dict[str, Any], side: str, round_no: int) -> dict[str, Any]:
    """그 회차가 지금도 주문 대상인지 보유 상태로만 확인한다 — 매도는 보유 중, 매수는 다음 회차.

    가격은 보지 않는다: 신호 가격과의 차이는 슬랙 확인 모달이 보여 주고 사람이 승인한다.
    """
    row = next((item for item in strategy["rounds"] if item["round"] == round_no), None)
    if row is None:
        raise BrokerApiError(f"{strategy['label']} 에 {round_no}호 회차가 없습니다.")
    orderable = row["held"] and round_no == strategy["status"]["held_count"] if side == "sell" else row["is_next"]
    if not orderable:
        word = "매도" if side == "sell" else "매수"
        raise RoundNotOrderableError(
            f"{strategy['label']} {round_no}호는 지금 {word}할 수 있는 회차가 아닙니다.",
            done=round_done(strategy, {"side": side, "round": round_no}),
        )
    return row


def find_duplicate_order(account_id: str, strategy_id: str, side: str) -> dict[str, Any] | None:
    """일부 체결로 보유 회차 번호가 바뀌어도 같은 방향의 진행 중인 주문을 찾는다."""
    return _orders().find_one(
        {"account_id": account_id, "strategy_id": strategy_id, "side": side, "status": {"$in": list(ACTIVE_STATUSES)}}
    )


def thread_has_filled_order(thread_ts: str) -> bool:
    """이 신호 스레드로 낸 주문 중 체결된 것이 있는지 — 있으면 체결 결과는 주문 결과 알림(`notify_finished_orders`)이 이미 알린다."""
    return _orders().find_one({"slack.thread_ts": thread_ts, "filled_quantity": {"$gt": 0}}, {"_id": 1}) is not None


def active_order_keys(strategy_ids: list[str]) -> set[tuple[str, int | str, str]]:
    """이 전략들의 아직 끝나지 않은 신호 주문 — (전략, 회차, 구분) 집합(화면 표시용)."""
    records = _orders().find(
        {
            "strategy_id": {"$in": strategy_ids},
            "status": {"$in": list(ACTIVE_STATUSES)},
            "$or": [{"round": {"$ne": None}}, {"arbitrary_id": {"$ne": None}}],
        },
        {"strategy_id": 1, "round": 1, "side": 1, "arbitrary_id": 1},
    )
    return {
        (record["strategy_id"], record.get("arbitrary_id") or record["round"], record["side"]) for record in records
    }


def active_order_count(account_id: str) -> int:
    """이 계좌의 아직 끝나지 않은 신호 주문 수(요약 알림용)."""
    return _orders().count_documents({"account_id": account_id, "status": {"$in": list(ACTIVE_STATUSES)}})


def _require_no_duplicate(account_id: str, strategy_id: str, side: str) -> None:
    found = find_duplicate_order(account_id, strategy_id, side)
    if found is not None:
        raise BrokerApiError(f"{'매도' if side == 'sell' else '매수'} 주문이 이미 있습니다({found['status']}).")


def _require_no_manual_order(strategy: dict[str, Any], side: str) -> None:
    """증권사 앱(MTS)에 같은 종목·같은 구분의 미체결 수동 주문이 걸려 있으면 막는다 — 중복 주문을 피하려면 그 주문부터 정리해야 한다."""
    found = [order for order in strategy["manual_orders"] if order["side"] == side]
    if found:
        word = "매도" if side == "sell" else "매수"
        detail = ", ".join(f"{order['price']:,.0f}원 {order['quantity']:,}주" for order in found)
        raise BrokerApiError(
            f"{strategy['label']} 에 증권사 앱에서 낸 미체결 {word} 주문이 있어 주문할 수 없습니다({detail}) — 먼저 정리하세요."
        )


def _require_order_allowed(account_id: str, env: str, amount: int, count: int) -> None:
    """운영 계좌의 주문 안전 장치 — 계좌의 API 주문 스위치, 정규장, 하루 주문 건수·금액 한도. 모의 계좌는 막지 않는다.

    ``count`` 는 이번에 낼 주문 건수(매도 후 재매수는 2건), ``amount`` 는 이번 주문의 순금액(종목 통화)이다 — 매수는 +, 매도는 −.
    금액 한도는 원화로 입력하며, 오늘 API 로 낸 주문의 순매수(매수 − 매도, 0 미만은 0)를 원화로 환산해 비교한다.
    매도는 한도를 쓰지 않고 오히려 되돌려 준다.
    """
    if env == ACCOUNT_ENV_MOCK:
        return
    doc = next(doc for doc in load_account_docs() if doc["account_id"] == account_id)
    if not doc.get("api_order_enabled"):
        raise BrokerApiError(
            f"'{doc['account_id']}' 계좌는 API 주문이 꺼져 있습니다 — 계좌 설정에서 주문 허용을 켜세요."
        )
    max_count, max_amount = doc.get("api_order_max_count"), doc.get("api_order_max_amount")
    if not max_count or not max_amount:
        raise BrokerApiError(f"'{doc['account_id']}' 계좌의 하루 주문 건수·금액 한도가 설정돼 있지 않습니다.")
    today = _now().strftime("%Y-%m-%d")
    used = list(
        _orders().find(
            {
                "account_id": account_id,
                "created_at": {"$gte": today},
                "status": {"$in": [*ACTIVE_STATUSES, "filled"]},
            }
        )
    )
    if len(used) + count > max_count:
        raise BrokerApiError(f"오늘 주문 건수 한도({max_count}건)를 넘습니다 — 이미 {len(used)}건입니다.")
    # 순매도 주문은 금액 한도를 넘길 수 없다 — 한도를 낮춘 뒤에도 매도는 나가야 한다(환율도 필요 없다).
    if amount <= 0:
        return
    rate = krw_rate(account_country(account_id))
    signed = sum(record["price"] * record["quantity"] * (1 if record["side"] == "buy" else -1) for record in used)
    used_krw = round(max(0, signed) * rate)
    amount_krw = round(amount * rate)
    if used_krw + amount_krw > max_amount:
        raise BrokerApiError(
            f"오늘 주문 금액 한도({max_amount:,}원)를 넘습니다 — 순매수 이미 {used_krw:,}원, 이번 {amount_krw:,}원입니다."
        )


def resolve_order(
    account_id: str,
    pool: str,
    strategy_id: str,
    side: str,
    round_no: int | None,
    *,
    fresh: bool,
    arbitrary_id: str | None = None,
) -> dict[str, Any]:
    """주문 내용 — 가격은 **그 순간의 상대 1호가 지정가**(매수는 매도 1호가, 매도는 매수 1호가), 매수 수량은 회차 금액과 지정가로, 매도 수량은 해당 회차의 실제 잔여 주수로 정한다.

    마지막 체결가로 내면 호가에 닿지 않아 체결이 안 되는 일이 잦다 — 상대 1호가는 바로 체결되는 가장 좋은 가격이다.
    사다리 지정가를 미리 걸지 않는다 — 사람이 버튼을 누르는 순간의 시장 호가로만 낸다.
    `round_no` 가 있으면(신호 주문) 그 회차가 지금도 주문 대상인지 확인하고, 비교용 `signal_price`(매도 목표·매수선,
    정해진 시각에 사는 1호는 None)를 함께 준다. `quote_label` 은 가격으로 쓴 호가의 이름이다.

    ``fresh`` 가 참이면 보유 회차를 증권사에서 새로 계산한다(수 초) — 실제 주문 직전 검증용이다. 거짓이면 배치가
    저장해 둔 것을 읽는다(1초 안팎) — 슬랙 모달은 3초 안에 열어야 해서 이쪽을 쓴다.

    운영 계좌는 전략의 운용 계좌로만 주문한다 — 다른 계좌의 체결로 회차를 세는 전략이라 엉뚱한 계좌에 사면
    회차가 어긋난다. 모의 계좌는 주문 경로 시험용이라 어느 전략 종목이든 받는다.
    """
    if side not in ("buy", "sell"):
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    strategy = _order_strategy(account_id, pool, strategy_id, fresh=fresh)
    _require_no_manual_order(strategy, side)
    signal_price = None
    target_round = round_no
    if arbitrary_id is not None:
        if side != "sell" or round_no is not None:
            raise BrokerApiError("임의 매도는 임의 매수 건만 지정할 수 있습니다.")
        row = next((lot for lot in strategy["arbitrary_lots"] if lot["id"] == arbitrary_id), None)
        if row is None:
            raise RoundNotOrderableError(f"{strategy['label']} 임의 매수 건의 잔여 수량이 없습니다.", done=True)
        signal_price = row["sell_limit"]
    elif side == "sell" and target_round is None:
        target_round = strategy["status"]["held_count"]
        if not target_round:
            raise RoundNotOrderableError(f"{strategy['label']}에 매도할 보유 회차가 없습니다.", done=True)
    if arbitrary_id is None:
        row = _require_round_state(strategy, side, target_round) if target_round is not None else None
    if round_no is not None:
        signal_price = row["sell_limit"] if side == "sell" else (None if row["close_entry"] else row["buy_limit"])
    venue = _order_venue(strategy)
    quote = fetch_best_quote(strategy["account_id"], strategy["ticker"], venue)
    label, price = ("매도 1호가", quote["ask"]) if side == "buy" else ("매수 1호가", quote["bid"])
    _require_valid_quote(strategy, label, price)
    if side == "buy":
        quantity = _buy_quantity(strategy, price, account_id)
    else:
        quantity = row["quantity"]
    return {
        "strategy_label": strategy["label"],
        "ticker": strategy["ticker"],
        "quantity": quantity,
        "price": price,
        "signal_price": signal_price,
        "quote_label": label,
        "venue": venue,
        "country": strategy["country"],
        "instrument_kind": strategy["instrument_kind"],
        "arbitrary_id": arbitrary_id,
    }


def _buy_quantity(strategy: dict[str, Any], price: int, account_id: str) -> int:
    size = strategy["config"]["round_size"]
    quantity = round_buy_quantity(size, price, krw_rate(strategy["country"]))
    if quantity == 0:
        raise BrokerApiError(
            f"{strategy['label']}: 회차 금액 {size:,}만원으로 "
            f"{format_money(price, strategy['country'])}에 1주도 살 수 없습니다."
        )
    if account_id == strategy["account_id"] and quantity * price > strategy["cash"]:
        raise BrokerApiError(f"{strategy['label']}: 계좌 현금으로 {quantity:,}주를 살 수 없습니다.")
    return quantity


def _order_venue(strategy: dict[str, Any]) -> str:
    """지금 이 전략 종목에 주문을 낼 거래소(국장 KRX·NXT, 미장은 종목의 거래소 코드) — 거래 시간 밖이면 주문할 수 없어 막는다."""
    try:
        return order_venue(strategy["country"], strategy["instrument_kind"], strategy.get("exchange"), now=_now())
    except ValueError as exc:
        raise BrokerApiError(f"{strategy['label']}: {exc}") from exc


def _require_valid_quote(strategy: dict[str, Any], label: str, price: int) -> None:
    if not is_valid_quote(price, kind=strategy["instrument_kind"], country=strategy["country"]):
        raise BrokerApiError(
            f"{strategy['ticker']} {label} {format_money(price, strategy['country'])}이 호가 단위와 맞지 않습니다."
        )


def _order_strategy(account_id: str, pool: str, strategy_id: str, *, fresh: bool) -> dict[str, Any]:
    """주문할 전략의 운용 현황 — 회차 원장을 못 만들었거나 전략이 없거나 다른 운영 계좌면 막는다."""
    views = load_views_where(lambda entry: entry["pool"] == pool, fresh=fresh)
    if strategy_id in views["errors"]:
        raise BrokerApiError(f"'{strategy_id}' 전략의 보유 회차를 확인할 수 없습니다: {views['errors'][strategy_id]}")
    strategy = next((row for row in views["strategies"] if row["strategy_id"] == strategy_id), None)
    if strategy is None:
        raise BrokerApiError(f"{pool} 종목풀에 '{strategy_id}' 전략이 없습니다.")
    if account_env(account_id) != ACCOUNT_ENV_MOCK and account_id != strategy["account_id"]:
        raise BrokerApiError(
            f"{strategy['label']} 전략의 운용 계좌는 '{strategy['account_id']}' 입니다 — 다른 운영 계좌로는 주문하지 않습니다."
        )
    return strategy


def resolve_swap(account_id: str, pool: str, strategy_id: str, round_no: int, *, fresh: bool) -> dict[str, Any]:
    """매도 후 재매수 주문 내용 — **같은 쪽 1호가**로 판다(매도 1호가)·산다(매수 1호가).

    그 회차를 팔면 다음 매수선(앞 회차 체결가 −간격%)이 매도가 이상이라 팔자마자 다시 사게 되는 경우에만 낸다
    (`rebuy_after_sell`). 매도는 회차의 남은 수량, 재매수는 회차 금액과 매수 1호가로 계산한다.
    """
    strategy = _order_strategy(account_id, pool, strategy_id, fresh=fresh)
    _require_no_manual_order(strategy, "sell")
    _require_no_manual_order(strategy, "buy")
    row = _require_round_state(strategy, "sell", round_no)
    if not row["rebuy_after_sell"]:
        raise RoundNotOrderableError(f"{strategy['label']} {round_no}호는 매도 후 재매수 대상이 아닙니다.", done=False)
    venue = _order_venue(strategy)
    quote = fetch_best_quote(strategy["account_id"], strategy["ticker"], venue)
    _require_valid_quote(strategy, "매도 1호가", quote["ask"])
    _require_valid_quote(strategy, "매수 1호가", quote["bid"])
    return {
        "strategy_label": strategy["label"],
        "ticker": strategy["ticker"],
        "sell_quantity": row["quantity"],
        "buy_quantity": _buy_quantity(strategy, quote["bid"], account_id),
        "sell_price": quote["ask"],
        "buy_price": quote["bid"],
        "signal_price": row["sell_limit"],
        "venue": venue,
        "country": strategy["country"],
        "instrument_kind": strategy["instrument_kind"],
    }


def place_order(
    account_id: str,
    pool: str,
    strategy_id: str,
    side: str,
    round_no: int | None,
    slack: dict[str, str] | None,
    *,
    arbitrary_id: str | None = None,
) -> dict[str, Any]:
    """전략 종목을 상대 1호가 지정가로 계좌에 주문하고 기록한다.

    `round_no` 는 신호 주문(신호 메시지의 회차 버튼)만 넘긴다 — 회차 보유 상태와 중복을 확인하고 기록에 남긴다.
    `slack`(채널·메시지 ts)은 슬랙 메시지로 낸 주문만 넘긴다 — 있으면 끝난 뒤 그 스레드로 알린다. 없으면(모달의
    직접 주문) 매도만 받는다: 매수는 슬랙의 사람 리뷰를 거치게 한다.
    """
    if slack is None and side != "sell":
        raise BrokerApiError("모달에서 직접 내는 주문은 매도만 됩니다 — 매수는 슬랙 주문 메시지로 내세요.")
    env = account_env(account_id)
    # 주문 직전이라 저장된 원장이 아니라 지금 증권사 기준으로 다시 계산한다.
    order = resolve_order(account_id, pool, strategy_id, side, round_no, fresh=True, arbitrary_id=arbitrary_id)
    _require_no_duplicate(account_id, strategy_id, side)
    sign = 1 if side == "buy" else -1
    _require_order_allowed(account_id, env, sign * order["price"] * order["quantity"], 1)
    base = _record_base(env, account_id, pool, strategy_id, round_no, order, slack)
    return {**_submit(base, side, order["price"]), "signal_price": order["signal_price"]}


def place_swap(account_id: str, pool: str, strategy_id: str, round_no: int, slack: dict[str, str]) -> dict[str, Any]:
    """매도 후 재매수 — 매도(매도 1호가)와 매수(매수 1호가) 지정가를 한 짝(`pair_id`)으로 낸다.

    매도가 접수되지 않으면 매수도 내지 않는다(오류). 매도는 접수됐는데 매수가 실패하면 매수 기록을 거부로 남기고
    `buy_error` 로 알린다 — 매도만 걸린 것은 보통의 매도 신호 주문과 같다.
    """
    from uuid import uuid4

    env = account_env(account_id)
    order = resolve_swap(account_id, pool, strategy_id, round_no, fresh=True)
    _require_no_duplicate(account_id, strategy_id, "sell")
    _require_no_duplicate(account_id, strategy_id, "buy")
    amount = order["buy_price"] * order["buy_quantity"] - order["sell_price"] * order["sell_quantity"]
    _require_order_allowed(account_id, env, amount, 2)
    base = _record_base(
        env, account_id, pool, strategy_id, round_no, {**order, "quantity": order["sell_quantity"]}, slack
    )
    base["pair_id"] = uuid4().hex
    sell = _submit(base, "sell", order["sell_price"])
    try:
        buy = _submit({**base, "quantity": order["buy_quantity"]}, "buy", order["buy_price"])
    except BrokerApiError as exc:
        return {"sell": sell, "buy": None, "buy_error": str(exc), "signal_price": order["signal_price"]}
    return {"sell": sell, "buy": buy, "buy_error": None, "signal_price": order["signal_price"]}


def _record_base(
    env: str,
    account_id: str,
    pool: str,
    strategy_id: str,
    round_no: int | None,
    order: dict[str, Any],
    slack: dict[str, str] | None,
) -> dict[str, Any]:
    """주문 기록의 공통 필드 — 구분·가격·상태는 `_submit` 이 채운다."""
    return {
        "env": env,
        "account_id": account_id,
        "pool": pool,
        "strategy_id": strategy_id,
        "round": round_no,
        "arbitrary_id": order.get("arbitrary_id"),
        "strategy_label": order["strategy_label"],
        "ticker": order["ticker"],
        "quantity": order["quantity"],
        # 주문을 낸 거래소 — 취소·만료·마감 경고가 이 거래소 기준이다.
        "venue": order["venue"],
        "country": order["country"],
        "instrument_kind": order["instrument_kind"],
        "slack": slack,
    }


def _submit(base: dict[str, Any], side: str, price: int) -> dict[str, Any]:
    """기록을 남기고 증권사에 지정가 주문을 보낸다 → 기록(직렬화). 실패는 기록에 남기고 그대로 올린다."""
    now = _now()
    doc = {
        **base,
        "side": side,
        "price": price,
        "order_no": None,
        "filled_quantity": 0,
        "average_fill_price": 0.0,
        "status": "submitting",
        "message": "",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "slack_notified": False,
    }
    orders = _orders()
    inserted = orders.insert_one(doc)
    try:
        placed = place_limit_order(base["account_id"], side, base["ticker"], base["quantity"], price, base["venue"])
    except OrderUncertainError as exc:
        # 주문이 들어갔는지 모른다 — 실패로 단정하지 않고 남겨 둔다(`refresh` 가 정리).
        orders.update_one(
            {"_id": inserted.inserted_id}, {"$set": {"message": str(exc), "updated_at": _now().isoformat()}}
        )
        raise
    except BrokerApiError as exc:
        orders.update_one(
            {"_id": inserted.inserted_id},
            {"$set": {"status": "rejected", "message": str(exc), "updated_at": _now().isoformat()}},
        )
        raise
    orders.update_one(
        {"_id": inserted.inserted_id},
        {
            "$set": {
                "status": "pending",
                "order_no": placed["order_no"],
                "message": placed["message"],
                "updated_at": _now().isoformat(),
            }
        },
    )
    return serialize_order(orders.find_one({"_id": inserted.inserted_id}))


def pair_records(pair_id: str) -> list[dict[str, Any]]:
    """매도 후 재매수 한 짝의 주문 기록 — 매도·매수 순."""
    return sorted(_orders().find({"pair_id": pair_id}), key=lambda record: record["side"] != "sell")


def read_trade_history_orders(account_id: str, start: date, as_of: date) -> list[dict[str, Any]]:
    """회차 체결 이력과 대조할 ``start`` ~ ``as_of`` 의 전략 주문을 읽는다."""
    return list(
        _orders().find(
            {
                "account_id": account_id,
                "created_at": {
                    "$gte": start.isoformat(),
                    "$lt": (as_of + timedelta(days=1)).isoformat(),
                },
                "round": {"$ne": None},
                "order_no": {"$ne": None},
            },
            {
                "created_at": 1,
                "ticker": 1,
                "order_no": 1,
                "round": 1,
                "side": 1,
                "pair_id": 1,
                "filled_quantity": 1,
                "average_fill_price": 1,
            },
        )
    )


def _current_order_no(record: dict[str, Any]) -> str:
    """취소 요청을 보낼 주문번호 — 증권사 앱에서 정정했으면 정정으로 새로 생긴 번호다."""
    for row in list_day_orders(record["account_id"], _order_date(record)):
        if row["order_no"] == record["order_no"]:
            return row["current_order_no"]
    raise BrokerApiError("증권사 주문내역에서 이 주문번호를 찾지 못했습니다.")


def cancel_record(order_id: str) -> dict[str, Any]:
    """대기중·부분체결 주문을 전량 취소하고 상태를 다시 읽는다."""
    orders = _orders()
    record = orders.find_one({"_id": ObjectId(order_id)})
    if record is None:
        raise BrokerApiError("주문 기록을 찾을 수 없습니다.")
    if record["status"] not in ("pending", "partial") or not record.get("order_no"):
        raise BrokerApiError("대기중이거나 부분체결인 주문만 취소할 수 있습니다.")
    cancelled = cancel_order(record["account_id"], _current_order_no(record), record["ticker"], record["venue"])
    orders.update_one(
        {"_id": record["_id"]},
        {"$set": {"message": cancelled["message"], "updated_at": _now().isoformat()}},
    )
    refresh_account_orders(record["account_id"])
    return serialize_order(orders.find_one({"_id": record["_id"]}))


def _order_date(record: dict[str, Any]) -> str:
    """증권사 주문내역의 주문일자 — 그 시장의 현지 날짜다."""
    return broker_order_date(record["country"], datetime.fromisoformat(record["created_at"]))


def _broker_status(record: dict[str, Any], broker_status: str) -> str:
    """증권사 상태를 기록 상태로 옮긴다 — 미체결 주문은 당일(세션) 주문이므로 접수한 세션 날짜가 지났거나 그 거래소의
    거래 시간이 끝났으면 만료로 본다(국장 KRX 는 정규장, NXT 는 애프터마켓 마감까지, 미장은 종목 종류의 거래 시간 밖)."""
    country, now = record["country"], _now()
    venue_open = is_order_open(country, record["venue"], record["instrument_kind"], now=now)
    placed = session_trade_date(country, datetime.fromisoformat(record["created_at"]))
    if broker_status in ("pending", "partial") and (placed < session_trade_date(country, now) or not venue_open):
        return "expired"
    return broker_status


def refresh_account_orders(account_id: str) -> None:
    """진행 중인 기록의 상태를 증권사 체결 조회로 갱신한다 — 조회는 주문일자마다 한 번이다."""
    orders = _orders()
    active = list(orders.find({"account_id": account_id, "status": {"$in": list(ACTIVE_STATUSES)}}))
    if not active:
        return
    broker_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for order_day in sorted({_order_date(record) for record in active}):
        for row in list_day_orders(account_id, order_day):
            broker_rows[(order_day, row["order_no"])] = row

    now = _now()
    for record in active:
        update: dict[str, Any] = {}
        if not record.get("order_no"):
            age = (now - datetime.fromisoformat(record["created_at"])).total_seconds()
            if age > _SUBMIT_TIMEOUT_SECONDS:
                update = {
                    "status": "rejected",
                    "message": "전송 결과를 확인하지 못했습니다 — 증권사 주문내역에서 직접 확인하세요.",
                }
        else:
            row = broker_rows.get((_order_date(record), record["order_no"]))
            if row is None:
                update = {"message": "증권사 주문내역에서 이 주문번호를 찾지 못했습니다."}
            else:
                update = {
                    "status": _broker_status(record, row["status"]),
                    "filled_quantity": row["filled_quantity"],
                    "average_fill_price": row["average_fill_price"],
                    "message": "",
                }
        if update:
            orders.update_one({"_id": record["_id"]}, {"$set": {**update, "updated_at": now.isoformat()}})


def serialize_order(record: dict[str, Any]) -> dict[str, Any]:
    created = datetime.fromisoformat(record["created_at"])
    same_day = created.date() == _now().date()
    return {
        "id": str(record["_id"]),
        "env": record["env"],
        "round": record["round"],
        "at": created.strftime("%H:%M:%S" if same_day else "%m-%d %H:%M"),
        "strategy_label": record["strategy_label"],
        "side": record["side"],
        "ticker": record["ticker"],
        "country": record["country"],
        "quantity": record["quantity"],
        "price": record["price"],
        "filled_quantity": record["filled_quantity"],
        "average_fill_price": record["average_fill_price"],
        # 전송 중에는 대기중으로 보인다 — 결과를 모르는 채 오래 남으면 refresh 가 거부로 정리한다.
        "status": "pending" if record["status"] == "submitting" else record["status"],
        "message": record.get("message") or "",
        # 매도 후 재매수 한 짝이면 그 짝의 id — 단독 주문은 None.
        "pair_id": record.get("pair_id"),
        "arbitrary_id": record.get("arbitrary_id"),
    }


def refreshed_order(account_id: str, order_id: str) -> dict[str, Any]:
    """증권사 체결 조회로 이 계좌의 진행 중 기록을 갱신한 뒤 주문 한 건의 현재 상태를 돌려준다."""
    refresh_account_orders(account_id)
    record = _orders().find_one({"_id": ObjectId(order_id)})
    if record is None:
        raise BrokerApiError("주문 기록을 찾을 수 없습니다.")
    return serialize_order(record)


def refresh_slack_order_accounts() -> int:
    """진행 중인 슬랙 주문이 있는 계좌의 상태를 증권사 체결 조회로 갱신하고, 갱신한 계좌 수를 돌려준다."""
    accounts = _orders().distinct("account_id", {"slack": {"$ne": None}, "status": {"$in": list(ACTIVE_STATUSES)}})
    for account_id in accounts:
        refresh_account_orders(account_id)
    return len(accounts)


def finished_slack_orders() -> list[dict[str, Any]]:
    """끝났는데 슬랙으로 아직 알리지 않은 슬랙 주문 — 증권사가 접수한 주문(주문번호 있음)만 대상이다.

    접수 전에 거부된 주문은 슬랙 처리부가 그 자리에서 알린다.
    """
    return list(
        _orders().find(
            {
                "slack": {"$ne": None},
                "slack_notified": False,
                "order_no": {"$ne": None},
                "status": {"$in": list(FINISHED_STATUSES)},
            }
        )
    )


def mark_slack_notified(order_id: Any) -> None:
    _orders().update_one({"_id": order_id}, {"$set": {"slack_notified": True}})


def _config():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db[_CONFIG_COLLECTION]


#: 신호 스레드의 주문 메시지 상태 — 진행 중 · 완료(체결 또는 이미 처리된 회차) · 실패(체결 없이 끝남) · 아직 손대지 않음.
MESSAGE_PENDING = "pending"
MESSAGE_DONE = "done"
MESSAGE_FAILED = "failed"
MESSAGE_OPEN = "open"


def register_signal_thread(channel: str, thread_ts: str, messages: list[dict[str, Any]]) -> None:
    """새 신호 메시지 스레드와 그 안의 주문 메시지(`{"ts", "request"}`)를 기록한다 — 반응·마감 정리가 이 기록을 쓴다.

    끝나지 않은 채 남은 기록은 슬랙 메시지 기록과 같은 보존 기간이 지나면 지운다(그 뒤로는 스레드를 찾을 수 없다).
    """
    today = _now().date()
    oldest = (today - timedelta(days=SIGNAL_RETENTION_DAYS)).isoformat()
    config = _config()
    config.update_one({"_id": _SETTINGS_KEY}, {"$pull": {"signal_threads": {"date": {"$lt": oldest}}}}, upsert=True)
    thread = {
        "channel": channel,
        "thread_ts": thread_ts,
        "messages": messages,
        "settled": [],
        "reaction": None,
        "date": today.isoformat(),
    }
    config.update_one({"_id": _SETTINGS_KEY}, {"$push": {"signal_threads": thread}}, upsert=True)


def signal_threads() -> list[dict[str, Any]]:
    doc = _config().find_one({"_id": _SETTINGS_KEY}, {"signal_threads": 1}) or {}
    return list(doc.get("signal_threads") or [])


def find_signal_thread(thread_ts: str) -> dict[str, Any] | None:
    return next((thread for thread in signal_threads() if thread["thread_ts"] == thread_ts), None)


def settle_signal_message(thread_ts: str, message_ts: str, outcome: str) -> None:
    """주문 없이 끝난 주문 메시지의 결과를 남긴다 — 이미 처리된 회차(완료) 또는 주문 불가·장 마감(실패)."""
    if outcome not in (MESSAGE_DONE, MESSAGE_FAILED):
        raise ValueError(f"알 수 없는 주문 메시지 결과입니다: {outcome}")
    _config().update_one(
        {"_id": _SETTINGS_KEY, "signal_threads.thread_ts": thread_ts},
        {"$push": {"signal_threads.$.settled": {"ts": message_ts, "outcome": outcome}}},
    )


def set_signal_thread_reaction(thread_ts: str, reaction: str | None) -> None:
    _config().update_one(
        {"_id": _SETTINGS_KEY, "signal_threads.thread_ts": thread_ts},
        {"$set": {"signal_threads.$.reaction": reaction}},
    )


def forget_signal_thread(thread_ts: str) -> None:
    """끝난 스레드는 더 추적할 일이 없어 목록에서 지운다."""
    _config().update_one({"_id": _SETTINGS_KEY}, {"$pull": {"signal_threads": {"thread_ts": thread_ts}}})


def _orders_outcome(orders: list[dict[str, Any]]) -> str:
    """한 주문 메시지의 마지막 주문(매도 후 재매수는 마지막 짝)의 결과 — 다시 주문하기로 낸 주문이 앞 주문을 대신한다."""
    latest = max(orders, key=lambda order: order["created_at"])
    pair_id = latest.get("pair_id")
    group = [order for order in orders if pair_id is not None and order.get("pair_id") == pair_id] or [latest]
    if any(order["status"] in ACTIVE_STATUSES for order in group):
        return MESSAGE_PENDING
    if pair_id is not None:
        return (
            MESSAGE_DONE if {o["side"] for o in group if o["status"] == "filled"} == {"buy", "sell"} else MESSAGE_FAILED
        )
    return MESSAGE_DONE if latest["status"] == "filled" else MESSAGE_FAILED


def signal_message_states(thread: dict[str, Any]) -> dict[str, str]:
    """스레드의 주문 메시지별 상태 — 주문이 있으면 주문 결과, 없으면 남긴 결과(`settled`), 둘 다 없으면 손대지 않음."""
    by_message: dict[str, list[dict[str, Any]]] = {}
    for order in _orders().find(
        {"slack.channel": thread["channel"], "slack.thread_ts": thread["thread_ts"]},
        {"slack.message_ts": 1, "status": 1, "side": 1, "pair_id": 1, "created_at": 1},
    ):
        by_message.setdefault(order["slack"]["message_ts"], []).append(order)
    settled = {item["ts"]: item["outcome"] for item in thread["settled"]}
    states = {}
    for message in thread["messages"]:
        ts = message["ts"]
        if ts in by_message:
            states[ts] = _orders_outcome(by_message[ts])
        else:
            states[ts] = settled.get(ts, MESSAGE_OPEN)
    return states


def reorderable_signal_messages(thread: dict[str, Any]) -> set[str]:
    """다시 주문하기 버튼이 남은 주문 메시지 — 마지막 주문이 하나도 체결되지 않고 취소된 단건 주문(`_finish` 와 같은 조건)."""
    latest: dict[str, dict[str, Any]] = {}
    for order in _orders().find(
        {"slack.channel": thread["channel"], "slack.thread_ts": thread["thread_ts"]},
        {"slack.message_ts": 1, "status": 1, "filled_quantity": 1, "pair_id": 1, "created_at": 1},
    ):
        ts = order["slack"]["message_ts"]
        if ts not in latest or order["created_at"] > latest[ts]["created_at"]:
            latest[ts] = order
    return {
        ts
        for ts, order in latest.items()
        if order.get("pair_id") is None and order["status"] == "canceled" and order["filled_quantity"] == 0
    }


def signal_thread_state(states: dict[str, str]) -> str | None:
    """신호 메시지 반응 하나 — 진행 중(⏳) · 완료(✅) · 실패(❌) · 없음(None) 중 하나만 정한다.

    진행 중인 주문이 있거나 일부만 끝났으면 진행 중, 전부 완료면 완료, 손댄 것 없이 열려 있으면 없음,
    열린 것 없이 끝났는데 완료가 아닌 메시지가 있으면 실패다.
    """
    values = set(states.values())
    if MESSAGE_PENDING in values:
        return MESSAGE_PENDING
    if values == {MESSAGE_DONE}:
        return MESSAGE_DONE
    if MESSAGE_OPEN in values:
        return MESSAGE_PENDING if values - {MESSAGE_OPEN} else None
    return MESSAGE_FAILED


def round_done(strategy: dict[str, Any], request: dict[str, Any]) -> bool:
    """주문 메시지가 노린 회차가 이미 처리됐는지 — 매수 N호는 보유 회차가 N 이상, 매도 N호는 N 미만, 임의 매도는 그 건이 없음.

    매도 후 재매수는 보유 회차 수로 끝났는지 알 수 없어 그 메시지의 주문 결과로만 판정한다(여기서는 처리되지 않은 것이다).
    """
    if request.get("rebuy"):
        return False
    if request.get("arbitrary_id") is not None:
        return all(lot["id"] != request["arbitrary_id"] for lot in strategy["arbitrary_lots"])
    held = int(strategy["status"]["held_count"])
    round_no = request["round"]
    if round_no is None:
        return request["side"] == "sell" and held == 0
    return held < round_no if request["side"] == "sell" else held >= round_no


def close_warning_slack_orders() -> list[dict[str, Any]]:
    """그 거래소 마감이 가까운데 아직 끝나지 않은 슬랙 주문 — 마감 경고를 아직 보내지 않은 것만(증권사가 접수한 주문만)."""
    now = _now()
    candidates = _orders().find(
        {
            "slack": {"$ne": None},
            "slack_close_warned": {"$ne": True},
            "order_no": {"$ne": None},
            "status": {"$in": list(ACTIVE_STATUSES)},
        }
    )
    return [
        record
        for record in candidates
        if closing_windows(record["country"], record["instrument_kind"], record["venue"], now=now) is not None
    ]


def close_warning_text(record: dict[str, Any]) -> str:
    """마감 경고 문구용 마감 시각과 거래소 — `15:30 KRX` · `20:00 ET`."""
    return close_text(record["country"], record["venue"], now=_now())


def mark_slack_close_warned(order_id: Any) -> None:
    _orders().update_one({"_id": order_id}, {"$set": {"slack_close_warned": True}})


def list_orders(account_id: str) -> list[dict[str, Any]]:
    """증권사 상태를 반영한 주문 기록 — 오늘 낸 주문과 아직 끝나지 않은 주문, 최신순."""
    account_env(account_id)
    refresh_account_orders(account_id)
    today = _now().strftime("%Y-%m-%d")
    cursor = (
        _orders()
        .find(
            {
                "account_id": account_id,
                "$or": [{"created_at": {"$gte": today}}, {"status": {"$in": list(ACTIVE_STATUSES)}}],
            }
        )
        .sort("created_at", -1)
    )
    return [serialize_order(record) for record in cursor]


def broker_orders_today(account_id: str) -> list[dict[str, Any]]:
    """증권사에 오늘 들어간 주문 전체 — 우리 기록과 대조하는 용도(우리가 안 보낸 주문도 보인다)."""
    return list_day_orders(account_id, _now().strftime("%Y%m%d"))


def order_sources(account_id: str, today_rows: list[dict[str, Any]]) -> dict[tuple[str, str], tuple[bool, str]]:
    """우리가 증권사에 낸 주문의 (날짜 YYYYMMDD, 주문번호) → 슬랙 신호 주문 여부 — 회차 원장이 체결의 출처를 가릴 때 쓴다.

    증권사가 주문을 받은 뒤 우리 기록에 주문번호가 적히기 전(`submitting`)에 원장이 계산되면 우리 주문이 수동으로 잡힌다 —
    그 틈에는 오늘 증권사 주문(``today_rows``) 중 종목·구분·수량·가격이 같은 것을 그 기록의 주문으로 본다.
    """
    records = _orders().find(
        {"account_id": account_id, "order_no": {"$ne": None}},
        {"order_no": 1, "round": 1, "created_at": 1, "arbitrary_id": 1, "country": 1},
    )
    sources = {
        (_order_date(record), str(int(record["order_no"]))): (
            record["round"] is not None or bool(record.get("arbitrary_id")),
            record.get("arbitrary_id") or REGULAR_SELL,
        )
        for record in records
    }
    today = _now()
    today_key = broker_order_date(account_country(account_id), today)
    submitting = _orders().find(
        {
            "account_id": account_id,
            "order_no": None,
            "status": "submitting",
            "created_at": {"$gte": f"{today:%Y-%m-%d}"},
        },
        {"ticker": 1, "side": 1, "quantity": 1, "price": 1, "round": 1, "arbitrary_id": 1},
    )
    for record in submitting:
        match = next(
            (
                row
                for row in today_rows
                if (today_key, row["order_no"]) not in sources
                and row["ticker"] == record["ticker"]
                and is_buy(row) == (record["side"] == "buy")
                and row["quantity"] == record["quantity"]
                and row["price"] == record["price"]
            ),
            None,
        )
        if match is not None:
            target = record.get("arbitrary_id")
            sources[(today_key, match["order_no"])] = (
                record["round"] is not None or bool(target),
                target or REGULAR_SELL,
            )
    return sources
