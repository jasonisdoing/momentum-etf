"""사고팔기 회차 원장 — 증권사 체결을 재생해 보유 회차별 실제 체결가·산 날짜·출처를 구한다.

계좌 평균단가에서 1호가를 거꾸로 계산하지 않는다. 사다리의 1호가는 1호를 실제로 산 체결가이고,
그 값은 증권사 일별 주문 조회의 체결을 처음부터 재생해 얻는다. 슬랙 신호 주문이든 증권사 앱(MTS)의 수동 주문이든
같은 조회에 나오므로 구분 없이 반영하고, 출처만 우리 주문 기록과 대조해 붙인다.

- 보유 수량은 증권사 잔고(실시간)가 정한다. 재생의 출발점은 둘 중 하나다.
  1. **기준점**(`strategy_trade_ledger_anchors`) — 마지막으로 확정한 전 거래일 마감 기준 회차(로트). 그 다음 날부터의
     체결만 더한다. 한 번 확정한 회차는 산 지 오래돼도 체결가가 이어진다. 재생 결과가 잔고와 다르면 기준점을 버리고 2로 간다.
  2. **최근 `LOOKBACK_DAYS` 일** — 그 전의 보유(`잔고 − 기간 순매수`)를 체결가를 모르는 기초분으로 맨 아래에 두고 재생한다.
     기초분이 모두 팔려 남은 보유가 전부 기간 안 매수로 설명되면 확정이고, 기초분이 남으면 체결가를 알 수 없어 오류다.
  재생 결과가 잔고와 다르면 평균단가로 대신하지 않고 `LedgerError` 로 알린다.
- 재생은 주식 단위다: 매수는 로트를 쌓고, 매도는 맨 위(마지막 산) 로트부터 뺀다 — 마지막 회차부터 팔리는 사다리 규칙이다.
  회차는 원주문(정정 포함) 단위 매수 로트 하나다.
- 지난 날짜의 조회 결과는 바뀌지 않아 DB(`strategy_trade_day_fills`)에 한 번만 받아 둔다. 오늘은 매번 조회한다.
- 계산은 증권사를 여러 번 불러 느리다(약 8초). 사고팔기 알림 배치가 `refresh_ledgers` 로 계산해 DB(`strategy_trade_ledgers`)에
  저장하고, 화면은 `read_ledgers` 로 저장된 값만 읽는다. 주문 직전 확인만 그 순간 다시 계산한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from services.broker_api_service import BrokerApiError
from services.broker_trade_service import fetch_order_balance, list_day_orders
from utils.db_manager import get_db_connection

KST = ZoneInfo("Asia/Seoul")
COLLECTION = "strategy_trade_day_fills"
SNAPSHOT_COLLECTION = "strategy_trade_ledgers"
ANCHOR_COLLECTION = "strategy_trade_ledger_anchors"
#: 기준점이 없을 때 재생하는 기간(달력일) — 그 전에 산 보유가 남아 있으면 체결가를 알 수 없어 오류다.
LOOKBACK_DAYS = 30
#: 기간 전 보유(체결가 모름)를 담는 로트의 출처 — 재생 뒤 남아 있으면 확정할 수 없다.
_SOURCE_UNKNOWN = "unknown"

SOURCE_SLACK = "slack"
SOURCE_MANUAL = "manual"

#: 캐시·재생에 필요한 체결 행의 필드.
_FILL_FIELDS = ("order_no", "ticker", "side_label", "filled_quantity", "average_fill_price")


class LedgerError(ValueError):
    """체결 이력으로 보유 회차를 맞출 수 없다 — 평균단가로 짐작해 대신하지 않고 이유를 알린다."""


@dataclass(frozen=True)
class Trade:
    """체결 한 건(주문 단위) — 같은 주문의 평균 체결가·체결 수량."""

    day: str
    buy: bool
    shares: int
    price: float
    source: str


@dataclass(frozen=True)
class Lot:
    shares: int
    price: float
    day: str
    source: str


@dataclass(frozen=True)
class Ledger:
    quantity: int
    lots: tuple[Lot, ...]
    #: 재생한 체결(시간순) — 기준점 또는 최근 기간부터다.
    trades: tuple[Trade, ...]
    #: 오늘 증권사에 걸려 있는 우리 기록에 없는(수동) 미체결 주문 — 구분·가격·남은 수량.
    manual_orders: tuple[dict[str, Any], ...]
    #: 이 원장을 계산한 시각(KST ISO) — 화면이 어느 시점 기준인지 보여 준다.
    computed_at: str

    @property
    def as_of(self) -> date:
        return datetime.fromisoformat(self.computed_at).astimezone(KST).date()


def is_buy(row: dict[str, Any]) -> bool:
    if "매수" in row["side_label"]:
        return True
    if "매도" in row["side_label"]:
        return False
    raise BrokerApiError(f"알 수 없는 주문 구분입니다: {row['side_label']}")


def _signed(row: dict[str, Any]) -> int:
    return row["filled_quantity"] if is_buy(row) else -row["filled_quantity"]


def _cache():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db[COLLECTION]


def _filled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row[field] for field in _FILL_FIELDS} for row in rows if row["filled_quantity"] > 0]


def _past_fills(account_id: str, day: date) -> list[dict[str, Any]]:
    """지난 날짜의 체결 행 — 한 번 받으면 DB 에 두고 다시 부르지 않는다(주말은 주문이 없다)."""
    if day.weekday() >= 5:
        return []
    key = f"{account_id}:{day:%Y%m%d}"
    cache = _cache()
    stored = cache.find_one({"_id": key})
    if stored is not None:
        return stored["rows"]
    rows = _filled(list_day_orders(account_id, f"{day:%Y%m%d}"))
    cache.replace_one({"_id": key}, {"_id": key, "account_id": account_id, "rows": rows}, upsert=True)
    return rows


def _day_trades(
    rows: list[dict[str, Any]], ticker: str, day: date, sources: dict[tuple[str, str], bool]
) -> list[Trade]:
    """그날 그 종목의 체결 — 주문번호 순(같은 날 안의 순서)."""
    trades: list[tuple[int, Trade]] = []
    for row in rows:
        if row["ticker"] != ticker:
            continue
        source = SOURCE_SLACK if sources.get((f"{day:%Y%m%d}", row["order_no"])) else SOURCE_MANUAL
        trade = Trade(f"{day:%Y-%m-%d}", is_buy(row), row["filled_quantity"], row["average_fill_price"], source)
        trades.append((int(row["order_no"]), trade))
    return [trade for _, trade in sorted(trades, key=lambda item: item[0])]


def read_cached_trade_days(account_id: str, as_of: date) -> dict[date, list[dict[str, Any]]]:
    """화면 이력에 쓸 최근 일별 체결 캐시만 읽는다."""
    start = as_of - timedelta(days=LOOKBACK_DAYS)
    prefix = f"{account_id}:"
    days = _cache().find({"_id": {"$gte": f"{prefix}{start:%Y%m%d}", "$lt": f"{prefix}{as_of:%Y%m%d}"}})
    return {date.fromisoformat(doc["_id"].removeprefix(prefix)): doc["rows"] for doc in days}


def round_history(ledger: Ledger, ticker: str, cached_days: dict[date, list[dict[str, Any]]]) -> list[Trade]:
    """원장에 이어지는 과거 체결을 붙여 회차를 재생한다. 캐시가 비면 그 이전으로 넘어가지 않는다."""
    start = ledger.as_of - timedelta(days=LOOKBACK_DAYS)
    start_iso = start.isoformat()
    trades = tuple(trade for trade in ledger.trades if trade.day >= start_iso)
    day = (date.fromisoformat(trades[0].day) if trades else ledger.as_of) - timedelta(days=1)
    opening = ledger.quantity - sum(trade.shares if trade.buy else -trade.shares for trade in trades)
    past: list[Trade] = []
    while day >= start:
        if day.weekday() < 5:
            rows = cached_days.get(day)
            if rows is None:
                break
            day_trades = _day_trades(rows, ticker, day, {})
            reversed_trades = list(reversed(day_trades))
            for trade in reversed_trades:
                opening += -trade.shares if trade.buy else trade.shares
                if opening < 0:
                    break
                past.append(trade)
            if opening < 0:
                break
        day -= timedelta(days=1)

    cycle: list[Trade] = []
    quantity = ledger.quantity
    for trade in (*reversed(trades), *past):
        if quantity == 0:
            break
        cycle.append(trade)
        quantity += -trade.shares if trade.buy else trade.shares
    return list(reversed(cycle))


def _replay(lots: list[Lot], trades: list[Trade]) -> list[Lot]:
    """체결을 재생해 보유 로트를 만든다 — 매수는 위에 쌓고, 매도는 맨 위(마지막) 로트부터 뺀다."""
    lots = list(lots)
    for trade in trades:
        if trade.buy:
            lots.append(Lot(trade.shares, trade.price, trade.day, trade.source))
            continue
        remaining = trade.shares
        while remaining > 0:
            if not lots:
                raise LedgerError(f"{trade.day} 매도가 보유 수량을 넘습니다 — 체결 이력이 맞지 않습니다.")
            top = lots[-1]
            take = min(remaining, top.shares)
            if take == top.shares:
                lots.pop()
            else:
                lots[-1] = replace(top, shares=top.shares - take)
            remaining -= take
    return lots


def _require_held(lots: list[Lot], quantity: int) -> None:
    held = sum(lot.shares for lot in lots)
    if held != quantity:
        raise LedgerError(f"체결 이력의 보유 {held:,}주가 증권사 잔고 {quantity:,}주와 다릅니다.")


class _History:
    """한 계좌의 날짜별 체결 — 지난 날짜는 캐시(`_past_fills`), 오늘은 이번 조회 결과."""

    def __init__(self, account_id: str, today: date, today_fills: list[dict[str, Any]], sources: dict) -> None:
        self.account_id, self.today, self.today_fills, self.sources = account_id, today, today_fills, sources

    def trades(self, ticker: str, start: date) -> tuple[list[Trade], list[Trade]]:
        """``start`` 부터 어제까지의 체결과 오늘 체결 — 시간순."""
        past: list[Trade] = []
        day = start
        while day < self.today:
            past.extend(_day_trades(_past_fills(self.account_id, day), ticker, day, self.sources))
            day += timedelta(days=1)
        return past, _day_trades(self.today_fills, ticker, self.today, self.sources)


def _from_anchor(
    anchor: dict[str, Any], history: _History, ticker: str, quantity: int
) -> tuple[list[Trade], list[Lot]]:
    """기준점(전 거래일 마감 로트)에서 이어 재생한다 → (체결, 어제 마감 로트). 잔고와 다르면 LedgerError."""
    past, today = history.trades(ticker, date.fromisoformat(anchor["as_of"]) + timedelta(days=1))
    closing = _replay([Lot(**lot) for lot in anchor["lots"]], past)
    _require_held(_replay(closing, today), quantity)
    return past + today, closing


def _from_window(history: _History, ticker: str, quantity: int) -> tuple[list[Trade], list[Lot]]:
    """최근 기간을 재생한다 — 기간 전 보유는 체결가를 모르는 기초분으로 맨 아래에 둔다.

    → (체결, 어제 마감 로트). 재생 뒤 기초분이 남으면 체결가를 알 수 없어 LedgerError.
    """
    past, today = history.trades(ticker, history.today - timedelta(days=LOOKBACK_DAYS))
    base = quantity - sum(trade.shares if trade.buy else -trade.shares for trade in past + today)
    if base < 0:
        raise LedgerError(
            f"최근 {LOOKBACK_DAYS}일 순매수가 증권사 잔고 {quantity:,}주보다 많습니다 — 체결 이력이 맞지 않습니다."
        )
    closing = _replay([Lot(base, 0.0, "", _SOURCE_UNKNOWN)] if base else [], past)
    lots = _replay(closing, today)
    _require_held(lots, quantity)
    unknown = sum(lot.shares for lot in lots if lot.source == _SOURCE_UNKNOWN)
    if unknown:
        raise LedgerError(
            f"최근 {LOOKBACK_DAYS}일보다 먼저 산 {unknown:,}주가 남아 있어 그 회차의 체결가를 알 수 없습니다 — "
            "그 주식이 팔리거나 전량 매도한 뒤부터 회차를 셉니다."
        )
    return past + today, closing


def _manual_orders(
    today_rows: list[dict[str, Any]], tickers: list[str], sources: dict[tuple[str, str], bool], today: date
) -> dict[str, list[dict[str, Any]]]:
    orders: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in tickers}
    for row in today_rows:
        if row["ticker"] not in orders or row["status"] not in ("pending", "partial"):
            continue
        if (f"{today:%Y%m%d}", row["order_no"]) in sources:
            continue
        orders[row["ticker"]].append(
            {
                "side": "buy" if is_buy(row) else "sell",
                "price": row["price"],
                "quantity": row["unfilled_quantity"],
            }
        )
    return orders


#: 잔고·체결 조회 사이에 체결이 끼어들면 다시 읽는 최대 횟수.
_SNAPSHOT_ATTEMPTS = 3


def _balance(account_id: str) -> dict[str, int]:
    return {row["ticker"]: int(row["quantity"]) for row in fetch_order_balance(account_id)["holdings"]}


def _consistent_snapshot(account_id: str, today: date) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """같은 시점의 잔고와 오늘 주문 목록 — 잔고 → 주문 목록 → 잔고 순으로 읽어 앞뒤 잔고가 같을 때만 쓴다.

    두 조회 사이에 체결이 끼어들면(주문 직후 배치 등) 잔고와 체결이 어긋나 회차를 잘못 센다. 앞뒤 잔고가 다르면
    그 사이에 체결이 있었던 것이므로 다시 읽는다. 끝내 안정되지 않으면 계산하지 않고 오류로 알린다.
    """
    before = _balance(account_id)
    for _ in range(_SNAPSHOT_ATTEMPTS):
        rows = list_day_orders(account_id, f"{today:%Y%m%d}") if today.weekday() < 5 else []
        after = _balance(account_id)
        if after == before:
            return after, rows
        before = after
    raise BrokerApiError(f"'{account_id}' 잔고가 조회 중에 계속 바뀌어(체결 진행 중) 회차 원장을 계산하지 못했습니다.")


def compute_ledgers(account_id: str, tickers: list[str]) -> dict[str, tuple[Ledger | LedgerError, dict | None]]:
    """계좌의 종목들(`tickers`)마다 회차 원장과 새 기준점(전 거래일 마감 로트, 못 정하면 None)을 만든다.

    체결 조회는 날짜당 한 번이라 종목 수와 무관하다. 체결 이력이 잔고와 맞지 않아 만들 수 없는 종목은 원장 대신
    `LedgerError` 를 값으로 돌려준다(다른 종목은 그대로 쓴다).
    """
    from utils.strategy_trade_orders import order_sources

    computed_at = datetime.now(KST).isoformat()
    today = datetime.now(KST).date()
    balance, today_rows = _consistent_snapshot(account_id, today)
    sources = order_sources(account_id, today_rows)
    history = _History(account_id, today, _filled(today_rows), sources)
    manual = _manual_orders(today_rows, tickers, sources, today)
    anchors = {doc["ticker"]: doc for doc in _anchors().find({"account_id": account_id, "ticker": {"$in": tickers}})}
    results: dict[str, tuple[Ledger | LedgerError, dict | None]] = {}
    for ticker in tickers:
        quantity = balance.get(ticker, 0)
        try:
            try:
                if ticker not in anchors:
                    raise LedgerError("기준점 없음")
                trades, closing = _from_anchor(anchors[ticker], history, ticker, quantity)
            except LedgerError:
                # 기준점이 없거나 잔고와 맞지 않으면(기준점 이후 이력 누락 등) 처음부터 다시 맞춘다.
                trades, closing = _from_window(history, ticker, quantity)
        except LedgerError as exc:
            results[ticker] = (exc, None)
            continue
        lots = _replay(closing, [trade for trade in trades if trade.day == f"{today:%Y-%m-%d}"])
        ledger = Ledger(
            quantity,
            tuple(lots),
            tuple(trades),
            tuple(manual[ticker]),
            computed_at,
        )
        # 어제 마감 로트에 체결가를 모르는 기초분이 남아 있으면 기준점으로 쓰지 않는다(오늘 팔려 확정된 경우).
        anchor = None
        if all(lot.source != _SOURCE_UNKNOWN for lot in closing):
            anchor = {"as_of": f"{today - timedelta(days=1):%Y-%m-%d}", "lots": [lot.__dict__ for lot in closing]}
        results[ticker] = (ledger, anchor)
    return results


def _anchors():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db[ANCHOR_COLLECTION]


def _snapshots():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db[SNAPSHOT_COLLECTION]


def _to_doc(account_id: str, ticker: str, ledger: Ledger | LedgerError) -> dict[str, Any]:
    base = {"_id": f"{account_id}:{ticker}", "account_id": account_id, "ticker": ticker}
    if isinstance(ledger, LedgerError):
        return {**base, "error": str(ledger)}
    return {
        **base,
        "error": None,
        "quantity": ledger.quantity,
        "lots": [lot.__dict__ for lot in ledger.lots],
        "trades": [trade.__dict__ for trade in ledger.trades],
        "manual_orders": list(ledger.manual_orders),
        "computed_at": ledger.computed_at,
    }


def _from_doc(doc: dict[str, Any]) -> Ledger | LedgerError:
    if doc["error"] is not None:
        return LedgerError(doc["error"])
    return Ledger(
        quantity=doc["quantity"],
        lots=tuple(Lot(**lot) for lot in doc["lots"]),
        trades=tuple(Trade(**trade) for trade in doc["trades"]),
        manual_orders=tuple(doc["manual_orders"]),
        computed_at=doc["computed_at"],
    )


def refresh_ledgers(account_id: str, tickers: list[str]) -> dict[str, Ledger | LedgerError]:
    """원장을 새로 계산해 저장하고, 확정한 종목은 기준점(전 거래일 마감 로트)도 갱신한다."""
    results = compute_ledgers(account_id, tickers)
    snapshots, anchors = _snapshots(), _anchors()
    for ticker, (ledger, anchor) in results.items():
        doc = _to_doc(account_id, ticker, ledger)
        snapshots.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        if anchor is not None:
            anchors.replace_one(
                {"_id": doc["_id"]},
                {"_id": doc["_id"], "account_id": account_id, "ticker": ticker, **anchor},
                upsert=True,
            )
    return {ticker: ledger for ticker, (ledger, _) in results.items()}


def forget_ledgers(account_id: str, keep: list[str]) -> None:
    """더 이상 전략이 없는 종목의 저장된 원장과 기준점을 지운다."""
    _snapshots().delete_many({"account_id": account_id, "ticker": {"$nin": keep}})
    _anchors().delete_many({"account_id": account_id, "ticker": {"$nin": keep}})


def read_ledgers(account_id: str, tickers: list[str]) -> dict[str, Ledger | LedgerError]:
    """배치가 저장해 둔 원장을 읽는다 — 아직 계산된 적이 없는 종목은 `LedgerError` 로 알린다."""
    stored = {doc["ticker"]: doc for doc in _snapshots().find({"account_id": account_id, "ticker": {"$in": tickers}})}
    return {
        ticker: _from_doc(stored[ticker])
        if ticker in stored
        else LedgerError("아직 계산되지 않았습니다 — 사고팔기 알림 배치가 한 번 돈 뒤에 표시됩니다.")
        for ticker in tickers
    }
