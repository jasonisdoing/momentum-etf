"""사고팔기 주문 기록 — 우리가 증권사에 보낸 주문과 그 상태(모의투자 시험용).

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
    cancel_order,
    list_day_orders,
    place_limit_order,
    require_mock_account,
)
from utils.db_manager import get_db_connection
from utils.strategy_trade_service import load_views_for_pools

KST = ZoneInfo("Asia/Seoul")
COLLECTION = "strategy_trade_orders"
ENV_MOCK = "mock"

#: 아직 끝나지 않은 상태 — 이 동안 체결 조회로 상태를 갱신한다.
ACTIVE_STATUSES = ("submitting", "pending", "partial")
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


def resolve_order(pool: str, strategy_id: str, side: str) -> dict[str, Any]:
    """주문 내용 — 전략 종목의 현재가를 호가 단위로 반올림한 지정가, 수량은 회차당 수량이다."""
    if side not in ("buy", "sell"):
        raise BrokerApiError(f"알 수 없는 주문 구분입니다: {side}")
    strategy = next(
        (row for row in load_views_for_pools([pool])["strategies"] if row["strategy_id"] == strategy_id), None
    )
    if strategy is None:
        raise BrokerApiError(f"{pool} 종목풀에 '{strategy_id}' 전략이 없습니다.")
    current = float(strategy["index"]["close"] or 0)
    if current <= 0:
        raise BrokerApiError(f"{strategy['label']} 의 현재가를 알 수 없습니다.")
    return {
        "strategy_label": strategy["label"],
        "ticker": strategy["ticker"],
        "quantity": int(strategy["config"]["round_quantity"]),
        "price": int(round_to_tick(current)),
    }


def place_order(account_no: str, pool: str, strategy_id: str, side: str) -> dict[str, Any]:
    """전략 종목을 현재가 지정가로 모의투자 계좌에 주문하고 기록한다."""
    require_mock_account(account_no)
    order = resolve_order(pool, strategy_id, side)
    now = _now()
    doc = {
        "env": ENV_MOCK,
        "account_no": account_no,
        "pool": pool,
        "strategy_id": strategy_id,
        "side": side,
        **order,
        "order_no": None,
        "filled_quantity": 0,
        "status": "submitting",
        "message": "",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
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
    return _serialize(orders.find_one({"_id": inserted.inserted_id}))


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
    return _serialize(orders.find_one({"_id": record["_id"]}))


def _order_date(record: dict[str, Any]) -> str:
    return str(record["created_at"])[:10].replace("-", "")


def refresh_account_orders(account_no: str) -> None:
    """진행 중인 기록의 상태를 증권사 체결 조회로 갱신한다 — 조회는 주문일자마다 한 번이다."""
    orders = _orders()
    active = list(orders.find({"env": ENV_MOCK, "account_no": account_no, "status": {"$in": list(ACTIVE_STATUSES)}}))
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
                update = {"status": row["status"], "filled_quantity": row["filled_quantity"], "message": ""}
        if update:
            orders.update_one({"_id": record["_id"]}, {"$set": {**update, "updated_at": now.isoformat()}})


def _serialize(record: dict[str, Any]) -> dict[str, Any]:
    created = datetime.fromisoformat(record["created_at"])
    same_day = created.date() == _now().date()
    return {
        "id": str(record["_id"]),
        "at": created.strftime("%H:%M:%S" if same_day else "%m-%d %H:%M"),
        "strategy_label": record["strategy_label"],
        "side": record["side"],
        "ticker": record["ticker"],
        "quantity": record["quantity"],
        "price": record["price"],
        "filled_quantity": record["filled_quantity"],
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
    return _serialize(record)


def list_orders(account_no: str) -> list[dict[str, Any]]:
    """증권사 상태를 반영한 주문 기록 — 오늘 낸 주문과 아직 끝나지 않은 주문, 최신순."""
    require_mock_account(account_no)
    refresh_account_orders(account_no)
    today = _now().strftime("%Y-%m-%d")
    cursor = (
        _orders()
        .find(
            {
                "env": ENV_MOCK,
                "account_no": account_no,
                "$or": [{"created_at": {"$gte": today}}, {"status": {"$in": list(ACTIVE_STATUSES)}}],
            }
        )
        .sort("created_at", -1)
    )
    return [_serialize(record) for record in cursor]


def broker_orders_today(account_no: str) -> list[dict[str, Any]]:
    """증권사에 오늘 들어간 주문 전체 — 우리 기록과 대조하는 용도(우리가 안 보낸 주문도 보인다)."""
    return list_day_orders(account_no, _now().strftime("%Y%m%d"))
