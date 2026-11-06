"""사고팔기 회차 원장 — 증권사 체결을 재생해 보유 회차별 실제 체결가·산 날짜·출처를 구한다.

계좌 평균단가에서 1호가를 거꾸로 계산하지 않는다. 사다리의 1호가는 1호를 실제로 산 체결가이고(백테스트와 같다),
그 값은 증권사 일별 주문 조회의 체결을 처음부터 재생해 얻는다. 슬랙 신호 주문이든 증권사 앱(MTS)의 수동 주문이든
같은 조회에 나오므로 구분 없이 반영하고, 출처만 우리 주문 기록과 대조해 붙인다.

- 보유 수량은 증권사 잔고(실시간)가 정한다. 오늘부터 거꾸로 가며 `보유 − 그 뒤 순매수 = 0` 이 되는 가장 최근 날
  (마지막으로 보유가 0이었던 날, 보유가 없어도 오늘 청산했다면 그 시작)을 찾고, 거기서부터 체결을 재생한다. 찾지 못하거나 재생 결과가 잔고와 다르면
  평균단가로 대신하지 않고 `LedgerError` 로 알린다.
- 재생은 주식 단위다: 매수는 로트를 쌓고, 매도는 맨 위(마지막 산) 로트부터 뺀다 — 마지막 회차부터 팔리는 사다리 규칙이다.
  회차는 로트를 회차당 수량씩 앞에서부터 끊은 것이다.
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
from services.broker_order_service import fetch_order_balance, list_day_orders
from utils.db_manager import get_db_connection

KST = ZoneInfo("Asia/Seoul")
COLLECTION = "strategy_trade_day_fills"
SNAPSHOT_COLLECTION = "strategy_trade_ledgers"
#: 마지막으로 보유가 0이었던 날을 이 기간(달력일) 안에서 찾는다 — 못 찾으면 오류다.
LOOKBACK_DAYS = 30

SOURCE_SLACK = "slack"
SOURCE_MANUAL = "manual"
SOURCE_MIXED = "mixed"

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
    trades: tuple[Trade, ...]
    #: 오늘 증권사에 걸려 있는 우리 기록에 없는(수동) 미체결 주문 — 구분·가격·남은 수량.
    manual_orders: tuple[dict[str, Any], ...]
    #: 이 원장을 계산한 시각(KST ISO) — 화면이 어느 시점 기준인지 보여 준다.
    computed_at: str

    @property
    def cost(self) -> float:
        """지금 보유의 매입 원가(원)."""
        return sum(lot.shares * lot.price for lot in self.lots)

    def round_fills(self, per_round: int) -> list[dict[str, Any]]:
        """보유 로트를 앞에서부터 회차당 수량씩 끊은 회차별 체결가(가중평균)·산 날짜(마지막 체결일)·출처.

        마지막 회차가 회차당 수량에 못 미치면 그 수량만 담은 부분 회차다(`shares`).
        """
        chunks: list[dict[str, Any]] = []
        current: list[Lot] = []
        filled = 0
        for lot in self.lots:
            remaining = lot.shares
            while remaining > 0:
                take = min(remaining, per_round - filled)
                current.append(replace(lot, shares=take))
                filled += take
                remaining -= take
                if filled == per_round:
                    chunks.append(_chunk(len(chunks) + 1, current))
                    current, filled = [], 0
        if current:
            chunks.append(_chunk(len(chunks) + 1, current))
        return chunks

    def round_events(self, per_round: int) -> list[dict[str, Any]]:
        """체결을 시간순으로, 회차 단위 이벤트로 쪼갠 것 — 매수는 보유 위에 쌓이는 회차, 매도는 맨 위 회차부터다."""
        events: list[dict[str, Any]] = []
        size = 0
        for trade in self.trades:
            if trade.buy:
                start, end = size, size + trade.shares
            else:
                start, end = size - trade.shares, size
            size = end if trade.buy else start
            overlaps = _round_overlaps(start, end, per_round)
            for round_no, shares in overlaps if trade.buy else reversed(overlaps):
                events.append(
                    {
                        "round": round_no,
                        "day": trade.day,
                        "source": trade.source,
                        "side": "buy" if trade.buy else "sell",
                        "price": trade.price,
                        "quantity": shares,
                    }
                )
        return events


def _chunk(round_no: int, lots: list[Lot]) -> dict[str, Any]:
    shares = sum(lot.shares for lot in lots)
    sources = {lot.source for lot in lots}
    return {
        "round": round_no,
        "shares": shares,
        "price": sum(lot.shares * lot.price for lot in lots) / shares,
        "day": max(lot.day for lot in lots),
        "source": sources.pop() if len(sources) == 1 else SOURCE_MIXED,
    }


def _round_overlaps(start: int, end: int, per_round: int) -> list[tuple[int, int]]:
    """주식 위치 [start, end) 가 걸치는 회차와 각 회차에서 차지하는 수량."""
    overlaps: list[tuple[int, int]] = []
    position = start
    while position < end:
        round_no = position // per_round + 1
        upto = min(end, round_no * per_round)
        overlaps.append((round_no, upto - position))
        position = upto
    return overlaps


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


def _past_fills(account_no: str, day: date) -> list[dict[str, Any]]:
    """지난 날짜의 체결 행 — 한 번 받으면 DB 에 두고 다시 부르지 않는다(주말은 주문이 없다)."""
    if day.weekday() >= 5:
        return []
    key = f"{account_no}:{day:%Y%m%d}"
    cache = _cache()
    stored = cache.find_one({"_id": key})
    if stored is not None:
        return stored["rows"]
    rows = _filled(list_day_orders(account_no, f"{day:%Y%m%d}"))
    cache.replace_one({"_id": key}, {"_id": key, "rows": rows}, upsert=True)
    return rows


def _window_trades(
    account_no: str,
    today: date,
    today_fills: list[dict[str, Any]],
    held: dict[str, int],
    sources: dict[tuple[str, str], bool],
) -> tuple[dict[str, list[Trade]], set[str]]:
    """종목마다 마지막 보유 0 이후의 체결을 시간순으로 모은다 — 오늘부터 하루씩 거꾸로 간다. 끝내 못 찾은 종목도 함께 돌려준다."""
    net = dict.fromkeys(held, 0)
    found: dict[str, list[tuple[str, int, Trade]]] = {ticker: [] for ticker in held}
    unresolved = set(held)
    for offset in range(LOOKBACK_DAYS + 1):
        if not unresolved:
            break
        day = today - timedelta(days=offset)
        rows = today_fills if offset == 0 else _past_fills(account_no, day)
        for ticker in sorted(unresolved):
            for row in (item for item in rows if item["ticker"] == ticker):
                net[ticker] += _signed(row)
                source = SOURCE_SLACK if sources.get((f"{day:%Y%m%d}", row["order_no"])) else SOURCE_MANUAL
                trade = Trade(f"{day:%Y-%m-%d}", is_buy(row), row["filled_quantity"], row["average_fill_price"], source)
                found[ticker].append((f"{day:%Y%m%d}", int(row["order_no"]), trade))
        unresolved = {ticker for ticker in unresolved if held[ticker] - net[ticker] != 0}
    ordered = {
        ticker: [trade for _, _, trade in sorted(items, key=lambda item: item[:2])] for ticker, items in found.items()
    }
    return ordered, unresolved


def _replay(trades: list[Trade], quantity: int) -> tuple[Lot, ...]:
    """체결을 재생해 지금 보유 로트를 만든다 — 매도는 맨 위(마지막) 로트부터 뺀다."""
    lots: list[Lot] = []
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
    held = sum(lot.shares for lot in lots)
    if held != quantity:
        raise LedgerError(f"체결 이력의 보유 {held:,}주가 증권사 잔고 {quantity:,}주와 다릅니다.")
    return tuple(lots)


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


def compute_ledgers(account_no: str, tickers: list[str]) -> dict[str, Ledger | LedgerError]:
    """증권사 계좌의 종목들(`tickers`)마다 회차 원장을 만든다 — 체결 조회는 날짜당 한 번이라 종목 수와 무관하다.

    체결 이력이 잔고와 맞지 않아 만들 수 없는 종목은 원장 대신 `LedgerError` 를 값으로 돌려준다(다른 종목은 그대로 쓴다).
    """
    from utils.strategy_trade_orders import order_sources

    computed_at = datetime.now(KST).isoformat()
    sources = order_sources(account_no)
    balance = {row["ticker"]: int(row["quantity"]) for row in fetch_order_balance(account_no)["holdings"]}
    today = datetime.now(KST).date()
    today_rows = list_day_orders(account_no, f"{today:%Y%m%d}") if today.weekday() < 5 else []
    held = {ticker: balance.get(ticker, 0) for ticker in tickers}
    trades, unresolved = _window_trades(account_no, today, _filled(today_rows), held, sources)
    manual = _manual_orders(today_rows, tickers, sources, today)
    ledgers: dict[str, Ledger | LedgerError] = {}
    for ticker in tickers:
        if ticker in unresolved:
            ledgers[ticker] = LedgerError(
                f"최근 {LOOKBACK_DAYS}일 체결 이력에서 보유가 0이던 날을 찾지 못했습니다 — 이력이 증권사 잔고 {held[ticker]:,}주와 맞지 않습니다."
            )
            continue
        try:
            lots = _replay(trades[ticker], held[ticker])
        except LedgerError as exc:
            ledgers[ticker] = exc
            continue
        ledgers[ticker] = Ledger(held[ticker], lots, tuple(trades[ticker]), tuple(manual[ticker]), computed_at)
    return ledgers


def _snapshots():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    return db[SNAPSHOT_COLLECTION]


def _to_doc(account_no: str, ticker: str, ledger: Ledger | LedgerError) -> dict[str, Any]:
    base = {"_id": f"{account_no}:{ticker}", "account_no": account_no, "ticker": ticker}
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


def refresh_ledgers(account_no: str, tickers: list[str]) -> dict[str, Ledger | LedgerError]:
    """원장을 새로 계산해 저장한다."""
    ledgers = compute_ledgers(account_no, tickers)
    snapshots = _snapshots()
    for ticker, ledger in ledgers.items():
        doc = _to_doc(account_no, ticker, ledger)
        snapshots.replace_one({"_id": doc["_id"]}, doc, upsert=True)
    return ledgers


def forget_ledgers(account_no: str, keep: list[str]) -> None:
    """더 이상 전략이 없는 종목의 저장된 원장을 지운다."""
    _snapshots().delete_many({"account_no": account_no, "ticker": {"$nin": keep}})


def read_ledgers(account_no: str, tickers: list[str]) -> dict[str, Ledger | LedgerError]:
    """배치가 저장해 둔 원장을 읽는다 — 아직 계산된 적이 없는 종목은 `LedgerError` 로 알린다."""
    stored = {doc["ticker"]: doc for doc in _snapshots().find({"account_no": account_no, "ticker": {"$in": tickers}})}
    return {
        ticker: _from_doc(stored[ticker])
        if ticker in stored
        else LedgerError("아직 계산되지 않았습니다 — 사고팔기 알림 배치가 한 번 돈 뒤에 표시됩니다.")
        for ticker in tickers
    }
