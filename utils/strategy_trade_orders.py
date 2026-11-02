"""사고팔기 주문 기록 — 우리가 증권사에 보낸 주문과 그 상태(API 주문).

시험 주문은 회차 사다리와 무관하다. 전략의 종목을 **지금 현재가를 지정가로** 사고판다 — 거래 경로만 시험한다.
기록은 **무엇을 보냈는지**(전략·방향·가격·수량)를 남기고, 상태(대기중·부분체결·완료·취소·거부)는
**증권사 체결 조회가 진실**이라 조회할 때마다 거기서 갱신한다. 우리가 상태를 추측하지 않는다.

- 가격·수량은 클라이언트가 보내지 않는다. 서버가 전략 운용 현황(`strategy_trade_service`)에서 현재가(호가 단위로
  반올림)와 회차당 수량을 읽는다 — 화면과 주문이 갈리지 않게 단일 소스를 지킨다.
- 주문번호 대응: 주문 응답의 시장주문번호(`mkt_orr_no`)를 체결 조회의 통합주문번호(`itg_orr_no`)와 같은
  번호로 본다(공식 예시가 같은 번호 공간을 쓴다). 조회 결과에서 못 찾으면 상태를 바꾸지 않고 그 사실을
  메시지로 남긴다 — 속성으로 짐작해 맞추지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from bson import ObjectId

from core.strategy.trade.rules import round_to_tick
from services.broker_api_service import BrokerApiError
from services.broker_order_service import (
    OrderUncertainError,
    account_env,
    cancel_order,
    list_day_orders,
    place_limit_order,
)
from utils.db_manager import get_db_connection
from utils.strategy_trade_service import load_views_for_pools

KST = ZoneInfo("Asia/Seoul")
COLLECTION = "strategy_trade_orders"

#: 아직 끝나지 않은 상태 — 이 동안 체결 조회로 상태를 갱신한다.
ACTIVE_STATUSES = ("submitting", "pending", "partial")
#: 끝난 상태 — 만료는 접수일이 지났는데도 증권사가 미체결로 답하는 당일 주문이다.
FINISHED_STATUSES = ("filled", "canceled", "rejected", "expired")
#: 전송 결과를 모른 채 이 시간이 지난 기록은 거부로 정리한다.
_SUBMIT_TIMEOUT_SECONDS = 120

_index_ready = False


def _orders():
    global _index_ready
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    collection = db[COLLECTION]
    if not _index_ready:
        collection.create_index([("account_no", 1), ("created_at", -1)])
        _index_ready = True
    return collection


def _now() -> datetime:
    return datetime.now(KST)


def _require_signal(strategy: dict[str, Any], side: str, round_no: int) -> dict[str, Any]:
    """그 회차의 신호가 지금도 유효해야 한다 — 매도는 보유 중이고 매도선에 닿았을 때, 매수는 다음 회차가 매수선에 닿았을 때."""
    row = next((item for item in strategy["rounds"] if item["round"] == round_no), None)
    if row is None:
        raise BrokerApiError(f"{strategy['label']} 에 {round_no}호 회차가 없습니다.")
    reached = (row["held"] and row["sell_reached"]) if side == "sell" else (row["is_next"] and row["buy_reached"])
    if not reached:
        word = "매도" if side == "sell" else "매수"
        raise BrokerApiError(f"{strategy['label']} {round_no}호 {word} 신호가 더 이상 유효하지 않습니다.")
    return row


def _require_no_duplicate(account_no: str, strategy_id: str, side: str, round_no: int) -> None:
    """같은 회차·구분의 주문이 진행 중이거나 오늘 체결됐으면 막는다 — 체결이 보유에 반영되기 전 같은 신호로 또 나가지 않게."""
    today = _now().strftime("%Y-%m-%d")
    found = _orders().find_one(
        {
            "account_no": account_no,
            "strategy_id": strategy_id,
            "side": side,
            "round": round_no,
            "$or": [
                {"status": {"$in": list(ACTIVE_STATUSES)}},
                {"status": "filled", "created_at": {"$gte": today}},
            ],
        }
    )
    if found is not None:
        raise BrokerApiError(
            f"{round_no}호 {'매도' if side == 'sell' else '매수'} 주문이 이미 있습니다({found['status']})."
        )


def resolve_order(pool: str, strategy_id: str, side: str, round_no: int | None) -> dict[str, Any]:
    """주문 내용 — 수량은 회차당 수량이다.

    가격: 직접 낸 주문은 현재가를 호가 단위로 반올림한 지정가(`price_kind` current). 신호 주문(`round_no`)은
    그 회차의 신호가 지금도 유효한지 확인하고, 전략이 정한 사다리 지정가(`limit`)로 낸다 — 백테스트와 같은
    체결 원칙이다(현재가가 지정가를 넘어선 쪽이면 호가에 바로 체결된다). 보유 0 인 1호의 장 마감 매수는
    사다리 가격에 닿지 않으므로 현재가 지정가다.
    """
    if side not in ("buy", "sell"):
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    strategy = next(
        (row for row in load_views_for_pools([pool])["strategies"] if row["strategy_id"] == strategy_id), None
    )
    if strategy is None:
        raise BrokerApiError(f"{pool} 종목풀에 '{strategy_id}' 전략이 없습니다.")
    row = _require_signal(strategy, side, round_no) if round_no is not None else None
    current = float(strategy["index"]["close"] or 0)
    if current <= 0:
        raise BrokerApiError(f"{strategy['label']} 의 현재가를 알 수 없습니다.")
    current_price = int(round_to_tick(current))
    if row is None or (side == "buy" and row["close_entry"]):
        price, price_kind = current_price, "current"
    else:
        price, price_kind = int(row["sell_limit"] if side == "sell" else row["buy_limit"]), "limit"
    return {
        "strategy_label": strategy["label"],
        "ticker": strategy["ticker"],
        "quantity": int(strategy["config"]["round_quantity"]),
        "price": price,
        "price_kind": price_kind,
        "current_price": current_price,
    }


def place_order(
    account_no: str, pool: str, strategy_id: str, side: str, round_no: int | None, slack: dict[str, str] | None
) -> dict[str, Any]:
    """전략 종목을 지정가로 계좌에 주문하고 기록한다 — 가격은 `resolve_order` 가 정한다.

    `round_no` 는 신호 주문(신호 메시지의 회차 버튼)만 넘긴다 — 신호 유효성과 중복을 확인하고 기록에 남긴다.
    `slack`(채널·메시지 ts)은 슬랙 메시지로 낸 주문만 넘긴다 — 있으면 끝난 뒤 그 스레드로 알린다.
    """
    env = account_env(account_no)
    order = resolve_order(pool, strategy_id, side, round_no)
    if round_no is not None:
        _require_no_duplicate(account_no, strategy_id, side, round_no)
    now = _now()
    doc = {
        "env": env,
        "account_no": account_no,
        "pool": pool,
        "strategy_id": strategy_id,
        "side": side,
        "round": round_no,
        **{key: order[key] for key in ("strategy_label", "ticker", "quantity", "price", "price_kind", "current_price")},
        "order_no": None,
        "filled_quantity": 0,
        "average_fill_price": 0.0,
        "status": "submitting",
        "message": "",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "slack": slack,
        "slack_notified": False,
    }
    orders = _orders()
    inserted = orders.insert_one(doc)
    try:
        placed = place_limit_order(account_no, side, order["ticker"], order["quantity"], order["price"])
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


def cancel_record(order_id: str) -> dict[str, Any]:
    """대기중·부분체결 주문을 전량 취소하고 상태를 다시 읽는다."""
    orders = _orders()
    record = orders.find_one({"_id": ObjectId(order_id)})
    if record is None:
        raise BrokerApiError("주문 기록을 찾을 수 없습니다.")
    if record["status"] not in ("pending", "partial") or not record.get("order_no"):
        raise BrokerApiError("대기중이거나 부분체결인 주문만 취소할 수 있습니다.")
    cancelled = cancel_order(record["account_no"], record["order_no"], record["ticker"])
    orders.update_one(
        {"_id": record["_id"]},
        {"$set": {"message": cancelled["message"], "updated_at": _now().isoformat()}},
    )
    refresh_account_orders(record["account_no"])
    return serialize_order(orders.find_one({"_id": record["_id"]}))


def _order_date(record: dict[str, Any]) -> str:
    return str(record["created_at"])[:10].replace("-", "")


def _broker_status(record: dict[str, Any], broker_status: str) -> str:
    """증권사 상태를 기록 상태로 옮긴다 — 접수일이 지난 미체결 주문은 당일 주문이므로 만료로 본다."""
    if broker_status in ("pending", "partial") and _order_date(record) < _now().strftime("%Y%m%d"):
        return "expired"
    return broker_status


def refresh_account_orders(account_no: str) -> None:
    """진행 중인 기록의 상태를 증권사 체결 조회로 갱신한다 — 조회는 주문일자마다 한 번이다."""
    orders = _orders()
    active = list(orders.find({"account_no": account_no, "status": {"$in": list(ACTIVE_STATUSES)}}))
    if not active:
        return
    broker_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for date in sorted({_order_date(record) for record in active}):
        for row in list_day_orders(account_no, date):
            broker_rows[(date, row["order_no"])] = row

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
        "quantity": record["quantity"],
        "price": record["price"],
        "filled_quantity": record["filled_quantity"],
        "average_fill_price": record["average_fill_price"],
        # 전송 중에는 대기중으로 보인다 — 결과를 모르는 채 오래 남으면 refresh 가 거부로 정리한다.
        "status": "pending" if record["status"] == "submitting" else record["status"],
        "message": record.get("message") or "",
    }


def refreshed_order(account_no: str, order_id: str) -> dict[str, Any]:
    """증권사 체결 조회로 이 계좌의 진행 중 기록을 갱신한 뒤 주문 한 건의 현재 상태를 돌려준다."""
    refresh_account_orders(account_no)
    record = _orders().find_one({"_id": ObjectId(order_id)})
    if record is None:
        raise BrokerApiError("주문 기록을 찾을 수 없습니다.")
    return serialize_order(record)


def refresh_slack_order_accounts() -> int:
    """진행 중인 슬랙 주문이 있는 계좌의 상태를 증권사 체결 조회로 갱신하고, 갱신한 계좌 수를 돌려준다."""
    accounts = _orders().distinct("account_no", {"slack": {"$ne": None}, "status": {"$in": list(ACTIVE_STATUSES)}})
    for account_no in accounts:
        refresh_account_orders(account_no)
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


def list_orders(account_no: str) -> list[dict[str, Any]]:
    """증권사 상태를 반영한 주문 기록 — 오늘 낸 주문과 아직 끝나지 않은 주문, 최신순."""
    account_env(account_no)
    refresh_account_orders(account_no)
    today = _now().strftime("%Y-%m-%d")
    cursor = (
        _orders()
        .find(
            {
                "account_no": account_no,
                "$or": [{"created_at": {"$gte": today}}, {"status": {"$in": list(ACTIVE_STATUSES)}}],
            }
        )
        .sort("created_at", -1)
    )
    return [serialize_order(record) for record in cursor]


def broker_orders_today(account_no: str) -> list[dict[str, Any]]:
    """증권사에 오늘 들어간 주문 전체 — 우리 기록과 대조하는 용도(우리가 안 보낸 주문도 보인다)."""
    return list_day_orders(account_no, _now().strftime("%Y%m%d"))
