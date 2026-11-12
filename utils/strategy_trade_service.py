"""전략 사고팔기 설정과 증권사 체결 원장을 연결해 운용 현황을 만든다."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from core.strategy.trade import rules as ladder
from services.broker_trade_service import resolve_us_exchange
from utils.logger import get_app_logger
from utils.strategy_trade_config import (
    DEFAULT_ROUND_SIZES,
    EDITABLE_CONFIG_KEYS,
    FIRST_ENTRY_CLOSE_TIMES,
    TRADE_COUNTRIES,
    TRADE_CURRENCIES,
    round_buy_quantity,
    validate_strategy_trade_config,
)
from utils.strategy_trade_ledger import (
    Ledger,
    LedgerError,
    Trade,
    forget_ledgers,
    read_cached_trade_days,
    read_ledgers,
    refresh_ledgers,
    round_history,
)

KST = ZoneInfo("Asia/Seoul")
_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"

logger = get_app_logger()


def _settings_collection():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 전략 사고팔기 설정을 읽을 수 없습니다.")
    return db[_CONFIG_COLLECTION]


def _load_settings_document() -> dict[str, Any]:
    return _settings_collection().find_one({"_id": _SETTINGS_KEY}) or {}


def load_settings() -> dict[str, Any]:
    """계좌별 슬랙 설정과 전략별 가격 출처·파라미터를 반환한다."""
    doc = _load_settings_document()
    stored = doc.get("strategies")
    if not isinstance(stored, dict):
        raise RuntimeError(
            f"저장된 전략 사고팔기 설정이 없습니다 "
            f"({_CONFIG_COLLECTION}.{_SETTINGS_KEY}.strategies 를 먼저 저장하세요)."
        )

    accounts: dict[str, dict[str, Any]] = {}
    for account_id, entry in (doc.get("accounts") or {}).items():
        if not isinstance(entry, dict) or not isinstance(entry.get("slack_enabled"), bool):
            raise RuntimeError(f"'{account_id}' 계좌의 사고팔기 설정이 올바르지 않습니다 — 화면에서 다시 저장하세요.")
        accounts[str(account_id)] = {"slack_enabled": entry["slack_enabled"], "groups": entry.get("groups", [])}

    strategies: dict[str, dict[str, Any]] = {}
    for sid, entry in stored.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"'{sid}' 전략 설정이 올바르지 않습니다 — 화면에서 다시 저장하세요.")
        pool = str(entry.get("pool") or "").strip()
        ticker = str(entry.get("ticker") or "").strip()
        name = str(entry.get("name") or "").strip()
        account_id = str(entry.get("account_id") or "").strip()
        if not pool or not ticker or not name or not account_id:
            raise RuntimeError(f"'{sid}' 전략의 종목풀·운용 계좌·종목이 없습니다 — 화면에서 골라 저장하세요.")
        try:
            strategies[sid] = {
                "pool": pool,
                # 운용 계좌 — 그 계좌의 체결로 회차를 세고 그 계좌로 주문한다.
                "account_id": account_id,
                "ticker": ticker,
                "name": name,
                # 미국 종목의 시세·주문 거래소 코드(OQ·NY·AX) — 국장 전략은 없다.
                "exchange": entry.get("exchange"),
                "sort_order": entry.get("sort_order"),
                **validate_strategy_trade_config({key: entry.get(key) for key in EDITABLE_CONFIG_KEYS}),
            }
        except ValueError as error:
            raise ValueError(f"'{sid}' 전략 설정이 올바르지 않습니다: {error}") from error

    return {"accounts": accounts, "strategies": strategies}


def _eligible_pool_kinds(countries: tuple[str, ...]) -> dict[str, ladder.InstrumentKind]:
    """``countries`` 의 활성 개별주·ETF 풀과 호가 계산에 사용할 종목 유형."""
    from utils.settings_loader import _load_pool_configs
    from utils.strategy_trade_config import is_eligible_pool

    return {
        str(config["ticker_type"]): "stock" if str(config["pool_kind"]).strip().lower() == "stock" else "etf"
        for config in _load_pool_configs()
        if is_eligible_pool(config, countries)
    }


def _pool_countries(countries: tuple[str, ...]) -> dict[str, ladder.Country]:
    """활성 개별주·ETF 풀의 국가 — 전략의 국가(호가 단위·통화·시간표)는 가격 출처 풀이 정한다."""
    from utils.settings_loader import _load_pool_configs
    from utils.strategy_trade_config import is_eligible_pool

    return {
        str(config["ticker_type"]): str(config["country_code"]).strip().lower()
        for config in _load_pool_configs()
        if is_eligible_pool(config, countries)
    }


def _pool_tickers(pool: str, kind: ladder.InstrumentKind) -> dict[str, str]:
    from utils.stock_list_io import get_etfs

    tickers: dict[str, str] = {}
    for item in get_etfs(pool):
        if kind == "stock" and (
            item.get("is_etf") is not False or str(item.get("market") or "").strip().upper() not in {"KOSPI", "KOSDAQ"}
        ):
            continue
        if kind == "etf" and item.get("is_etf") is False:
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        name = str(item.get("name") or "").strip()
        if ticker and name:
            tickers[ticker] = name
    return tickers


def _ticker_options(pools: dict[str, ladder.InstrumentKind]) -> dict[str, dict[str, Any]]:
    """종목을 티커로 합치되, 풀마다 종목 유형이 충돌하면 등록 대상에서 뺀다."""
    options: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for pool, kind in sorted(pools.items()):
        for ticker, name in _pool_tickers(pool, kind).items():
            option = options.setdefault(ticker, {"ticker": ticker, "name": name, "pools": [], "kind": kind})
            if option["kind"] != kind:
                conflicts.add(ticker)
            option["pools"].append(pool)
    return {ticker: option for ticker, option in options.items() if ticker not in conflicts}


def _validated_target(
    account_id: str,
    config: dict[str, Any],
    current: dict[str, Any] | None,
    options: dict[str, dict[str, Any]],
    country: str,
) -> dict[str, Any]:
    ticker = str(config.get("ticker") or "").strip().upper()
    reselect = config.get("reselect_source", False)
    if not isinstance(reselect, bool):
        raise ValueError("reselect_source는 참·거짓이어야 합니다.")
    if current is not None and ticker == current["ticker"] and not reselect:
        pool, name = current["pool"], current["name"]
    else:
        option = options.get(ticker)
        if option is None:
            raise ValueError(f"이 계좌 국가에서 사고팔기에 등록할 수 있는 개별주·ETF가 아닙니다: {ticker}")
        candidates = option["pools"]
        # 기존 출처가 여전히 유효하면 같은 종목의 기준가를 바꾸지 않는다.
        pool = (
            current["pool"]
            if current and ticker == current["ticker"] and current["pool"] in candidates
            else candidates[0]
        )
        name = option["name"]
    target = {
        "pool": pool,
        "account_id": account_id,
        "ticker": ticker,
        "name": name,
        **validate_strategy_trade_config(config),
    }
    if country == "us":
        # 거래소는 종목을 고를 때 메리츠 시세로 한 번 판정해 저장한다(같은 종목이면 저장된 값을 쓴다).
        same = current is not None and ticker == current["ticker"] and current.get("exchange")
        target["exchange"] = current["exchange"] if same else resolve_us_exchange(account_id, ticker)
    return target


def _sent_log_unset(sent_log: dict[str, Any], strategy_id: str) -> dict[str, str]:
    """그 전략의 알림 중복 방지 기록을 지우는 $unset 항목."""
    return {f"sent_log.{key}": "" for key in sent_log if str(key).startswith(f"{strategy_id}:")}


def _normalize_trade_groups(groups: Any) -> list[dict[str, str | None]]:
    if not isinstance(groups, list):
        raise ValueError("그룹 목록 형식이 올바르지 않습니다.")
    result: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for row in groups:
        if not isinstance(row, dict):
            raise ValueError("그룹 행의 형식이 올바르지 않습니다.")
        group_id, name, anchor = row.get("id"), row.get("name"), row.get("before_ticker")
        if not isinstance(group_id, str) or not group_id.strip() or group_id.strip() in seen:
            raise ValueError("그룹 식별자가 비어 있거나 중복됐습니다.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("그룹 이름을 입력하세요.")
        if anchor is not None and (not isinstance(anchor, str) or not anchor.strip()):
            raise ValueError("그룹 위치의 종목코드가 올바르지 않습니다.")
        seen.add(group_id.strip())
        result.append(
            {"id": group_id.strip(), "name": name.strip(), "before_ticker": anchor.strip().upper() if anchor else None}
        )
    return result


def save_account_trade(account_id: str, slack_enabled: Any, strategies: Any, groups: Any = None) -> None:
    """계좌의 슬랙 설정과 전략 목록을 검증한 뒤 한 번에 저장한다."""
    from uuid import uuid4

    account = next((item for item in _trade_account_docs() if item["account_id"] == account_id), None)
    if account is None:
        raise ValueError(f"'{account_id}' 는 사고팔기를 운영할 수 있는 계좌가 아닙니다(증권사 API 연동 필요).")
    country = str(account["country_code"]).strip().lower()
    if not isinstance(slack_enabled, bool):
        raise ValueError("slack_enabled 는 참·거짓이어야 합니다.")
    if not isinstance(strategies, list):
        raise ValueError("strategies 형식이 올바르지 않습니다.")
    options = _ticker_options(_eligible_pool_kinds((country,)))
    current = {sid: entry for sid, entry in load_settings()["strategies"].items() if entry["account_id"] == account_id}
    taken: set[str] = set()
    seen_ids: set[str] = set()
    planned: dict[str, dict[str, Any]] = {}
    for sort_order, row in enumerate(strategies):
        if not isinstance(row, dict):
            raise ValueError("전략 행의 형식이 올바르지 않습니다.")
        sid = str(row.get("strategy_id") or "").strip()
        if sid and (sid not in current or sid in seen_ids):
            raise ValueError(f"이 계좌의 전략이 아니거나 중복된 전략입니다: {sid}")
        if sid:
            seen_ids.add(sid)
        previous = current.get(sid)
        target = _validated_target(account_id, row, previous, options, country)
        if target["ticker"] in taken:
            raise ValueError(f"{target['name']}({target['ticker']}) 전략이 이 계좌에 이미 있습니다.")
        taken.add(target["ticker"])
        # 종목이 바뀌면 이전 주문 버튼이 새 종목을 가리키지 않도록 별도 전략으로 저장한다.
        target_id = sid if previous and previous["ticker"] == target["ticker"] else f"s{uuid4().hex[:8]}"
        planned[target_id] = {**target, "sort_order": sort_order}

    collection = _settings_collection()
    stored = collection.find_one({"_id": _SETTINGS_KEY}, {"sent_log": 1, "accounts": 1}) or {}
    sent_log = stored.get("sent_log") or {}
    account_settings: dict[str, Any] = {"slack_enabled": slack_enabled}
    if groups is None:
        groups = ((stored.get("accounts") or {}).get(account_id) or {}).get("groups", [])
    account_settings["groups"] = _normalize_trade_groups(groups)
    # 공유 DB에서 구버전 배치가 읽는 동안만 기존 계좌 형식을 유지한다.
    if any("pools" in account for account in (stored.get("accounts") or {}).values()):
        account_settings["pools"] = sorted({entry["pool"] for entry in planned.values()})
    unset: dict[str, str] = {}
    for sid, entry in current.items():
        if sid not in planned:
            unset.update({f"strategies.{sid}": "", **_sent_log_unset(sent_log, sid)})
        elif planned[sid]["pool"] != entry["pool"]:
            # 이전 가격 출처의 알림 중복 방지 기록은 새 출처에 적용하지 않는다.
            unset.update(_sent_log_unset(sent_log, sid))
    update: dict[str, Any] = {
        "$set": {
            "updated_at": datetime.now().isoformat(),
            f"accounts.{account_id}": account_settings,
            **{f"strategies.{sid}": target for sid, target in planned.items()},
        }
    }
    if unset:
        update["$unset"] = unset
    collection.update_one({"_id": _SETTINGS_KEY}, update, upsert=True)


def _load_account_state(account_id: str, country: ladder.Country) -> dict[str, Any]:
    """계좌 보유 종목의 현재가와 현금 (전 전략 공용, 1회 조회).

    현재가는 ``load_real_holdings_table`` 이 실시간 시세로 채운 값이다. 수량·평균단가는 쓰지 않는다 —
    보유는 증권사 체결 이력(회차 원장)이 정한다. 현금은 잔고 동기화가 저장한 값(자산 화면과 같은 값)이다 —
    ``cash`` 는 매수에 쓰는 계좌 통화(국장 원화·미장 달러) 예수금이다.
    """
    from utils.broker_balance_sync import stored_cash_map
    from utils.portfolio_io import load_portfolio_master, load_real_holdings_table

    master = load_portfolio_master(account_id)
    if master is None:
        raise ValueError(f"'{account_id}' 계좌의 잔고가 저장돼 있지 않습니다 — 잔고 동기화를 먼저 돌리세요.")
    currency = TRADE_CURRENCIES[country]
    cash = stored_cash_map(master, [currency])[currency]
    frame = load_real_holdings_table(account_id)
    closes: dict[str, float] = {}
    if frame is not None and not frame.empty:
        missing = {"티커", "현재가"} - set(frame.columns)
        if missing:
            raise RuntimeError(f"보유 테이블에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")
        for _, row in frame.iterrows():
            ticker = str(row["티커"]).strip()
            if ticker:
                closes[ticker] = float(row["현재가"] or 0)
    return {
        "closes": closes,
        "cash": cash,
    }


def _load_unheld_closes(tickers: list[str], pool: str, country: ladder.Country) -> dict[str, float]:
    """미보유 종목의 현재가 — 실시간 시세 우선, 실패분은 가격 캐시 폴백.

    1호 진입 판정에는 아직 안 산 종목의 현재가가 필요하다(없으면 매수 신호가
    영원히 안 뜬다). 실시간(pools-rank 와 같은 소스)을 1순위로 쓰고, 캐시 폴백은
    전략 종목이 든 풀(``pool``)을 본다.
    """
    closes: dict[str, float] = {}
    if not tickers:
        return closes

    try:
        from services.price_service import get_realtime_snapshot

        snapshot = get_realtime_snapshot(country, tickers)
        for ticker in tickers:
            value = (snapshot.get(ticker) or {}).get("nowVal")
            if isinstance(value, (int, float)) and float(value) > 0:
                closes[ticker] = float(value)
    except Exception:
        pass  # 실시간 실패는 캐시 폴백으로 이어진다 — 아래에서 남은 티커만 채운다.

    remaining = [t for t in tickers if t not in closes]
    if remaining:
        import pandas as pd

        from utils.cache_utils import load_cached_frames_bulk_from_ticker_types

        frames = load_cached_frames_bulk_from_ticker_types([pool], remaining)
        for ticker in remaining:
            frame = frames.get(ticker)
            if frame is None or frame.empty or "Close" not in frame.columns:
                continue
            series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if not series.empty:
                closes[ticker] = float(series.iloc[-1])
    return closes


# 회차 표의 저점 기준선 기간(개월).
RECENT_PRICE_MONTHS = 12


def _load_ticker_status(
    pool: str,
    ticker: str,
    name: str,
    trigger_pct: float,
    live_close: float | None,
    *,
    kind: ladder.InstrumentKind,
    country: ladder.Country,
) -> dict[str, Any]:
    """판정 기준 시세 — 이 전략 종목 자신의 현재가·전일 종가·1호 진입선·최근 고점과 저점.

    (이전에는 코스피/코스닥 지수를 썼지만, 이제 판정·표시 모두 자기 종목 가격이다)
    """
    import pandas as pd

    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types

    frames = load_cached_frames_bulk_from_ticker_types([pool], [ticker])
    frame = frames.get(ticker)
    if frame is None or frame.empty or "Close" not in frame.columns:
        raise RuntimeError(f"{name}({ticker}) 가격 캐시를 불러올 수 없습니다.")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 2:
        raise RuntimeError(f"{name}({ticker}) 종가 데이터가 부족합니다.")

    # 1호 진입 판정의 기준은 **직전 거래일 종가**다. 현재가에 -간격% 를 걸면 값이
    # 내릴 때마다 기준선도 같이 내려가 영영 닿지 않는다(움직이는 과녁).
    today = pd.Timestamp.now().normalize()
    prev_close = float(close.iloc[-2]) if close.index[-1].normalize() >= today else float(close.iloc[-1])
    if prev_close <= 0:
        raise RuntimeError(f"{name}({ticker}) 의 직전 거래일 종가가 올바르지 않습니다: {prev_close}")

    current = float(live_close) if live_close else float(close.iloc[-1])
    change_pct = (current / prev_close - 1.0) * 100.0

    # 확정 종가의 같은 기간에서 고점·저점을 구하고 현재가가 범위를 벗어나면 오늘 날짜를 쓴다.
    recent = close[close.index >= close.index[-1] - pd.DateOffset(months=RECENT_PRICE_MONTHS)]
    low_value, low_date = float(recent.min()), recent.idxmin()
    high_value, high_date = float(recent.max()), recent.idxmax()
    if current < low_value:
        low_value, low_date = current, today
    if current > high_value:
        high_value, high_date = current, today

    return {
        "name": name,
        "as_of": close.index[-1].strftime("%Y-%m-%d"),
        "close": ladder.round_price(current, country),
        "prev_close": ladder.round_price(prev_close, country),
        "change_pct": round(change_pct, 2),
        # 1호 진입선 — 전일 종가 대비 -간격%, 호가 단위 반올림.
        "buy_trigger": ladder.first_buy_limit(prev_close, float(trigger_pct), kind=kind, country=country),
        "recent_low": ladder.round_price(low_value, country),
        "recent_low_date": low_date.strftime("%Y-%m-%d"),
        "recent_low_label": f"{RECENT_PRICE_MONTHS}개월 저점",
        "recent_high": ladder.round_price(high_value, country),
        "recent_high_date": high_date.strftime("%Y-%m-%d"),
        "recent_high_label": f"{RECENT_PRICE_MONTHS}개월 고점",
    }


def _first_entry_close_window(
    kind: ladder.InstrumentKind, country: ladder.Country, now: datetime | None = None
) -> bool:
    """지금이 1호 장 마감 매수 시간인지 — 그 시장·종목 종류의 ``FIRST_ENTRY_CLOSE_TIMES`` 시각 이후이고 아직 거래 중일 때.

    국장 ETF 는 15:00~15:30(정규장), 개별주는 19:30~20:00(NXT 애프터마켓), 미장은 19:30~20:00 ET(애프터마켓)이다.
    거래일·거래 시간 판정은 시장·종목 종류별 시간표가 한다.
    """
    from zoneinfo import ZoneInfo

    from config import MARKET_SCHEDULES
    from utils.market_session import is_instrument_trading_time

    zone = ZoneInfo(MARKET_SCHEDULES[country]["timezone"])
    now_local = (now or datetime.now(zone)).astimezone(zone)
    return now_local.time() >= FIRST_ENTRY_CLOSE_TIMES[country][kind] and is_instrument_trading_time(
        country, kind, now=now_local
    )


def _paired_trade_rounds(trades: list[Trade], orders: list[dict[str, Any]]) -> dict[int, int]:
    """같은 날 매도·재매수가 모두 확인된 주문 쌍의 체결 인덱스를 회차에 연결한다."""
    by_order = {}
    by_fill = defaultdict(list)
    for order in orders:
        day = order["created_at"][:10]
        buy = order["side"] == "buy"
        by_order[(day, str(int(order["order_no"])), buy)] = order
        by_fill[(day, buy, order.get("filled_quantity"), order.get("average_fill_price"))].append(order)

    fill_counts = Counter((trade.day, trade.buy, trade.shares, trade.price) for trade in trades)
    paired = defaultdict(list)
    for index, trade in enumerate(trades):
        if trade.order_no is not None:
            order = by_order.get((trade.day, str(int(trade.order_no)), trade.buy))
        else:
            # 기존 원장에는 주문번호가 없다. 양쪽 기록에서 유일한 슬랙 체결만 연결한다.
            signature = (trade.day, trade.buy, trade.shares, trade.price)
            matches = by_fill.get(signature, [])
            order = matches[0] if trade.source == "slack" and fill_counts[signature] == len(matches) == 1 else None
        if order is not None and (pair_id := order.get("pair_id")):
            paired[(trade.day, pair_id, order["round"])].append(index)

    return {
        index: round_no
        for (_, _, round_no), indices in paired.items()
        if len(indices) == 2 and {trades[index].buy for index in indices} == {True, False}
        for index in indices
    }


def _attach_trade_history(
    view: dict[str, Any], ledger: Ledger, cached_days: dict[date, list[dict[str, Any]]], orders: list[dict[str, Any]]
) -> None:
    """사다리 위치별 가격(`position_prices`)과 날짜별 매수·매도를 체결가의 위치마다 체결 순서대로 붙인다.

    ``trades_by_position`` = {위치: {날짜: ["buy", "sell", …]}}. 위치는 지금 1호 체결가 기준으로 체결가에 가장 가까운
    칸이다. 같은 날 체결된 매도·재매수 주문 쌍은 해당 회차의 칸에 함께 표시한다.
    지금 사이클의 체결만 담는다 — 1호를 마지막으로 판 체결(그때 보유가 모두 비었다) 뒤부터다. 보유가 없으면 비어 있다.
    """
    trades: dict[int, dict[str, list[str]]] = {}
    view["trades_by_position"] = trades
    first = next((row for row in view["rounds"] if row["round"] == 1 and row["held"]), None)
    if first is None:
        # 보유가 없으면 사다리 기준(1호 체결가)이 없다 — 위치 가격은 회차의 1호 진입 미리보기 지정가다.
        view["position_prices"] = {row["position"]: row["buy_limit"] for row in view["rounds"]}
        return
    trigger_pct = float(view["config"]["trigger_pct"])
    history = round_history(ledger, view["ticker"], cached_days)
    paired_rounds = _paired_trade_rounds(history, orders)
    round_positions = {row["round"]: row["position"] for row in view["rounds"]}
    for index, trade in enumerate(history):
        position = round_positions.get(paired_rounds.get(index))
        if position is None:
            position = ladder.ladder_position(trade.price, first["fill_price"], trigger_pct)
        badges = trades.setdefault(position, {}).setdefault(trade.day, [])
        side = "buy" if trade.buy else "sell"
        if trade.shares > trade.arbitrary_shares:
            badges.append(side)
        if trade.arbitrary_shares:
            badges.append(f"arbitrary_{side}")
    # 위치 가격 — 회차·매매가 있는 위치 범위 전체(빈 칸 포함). 1호 체결가 × (1 − 간격%)^(k−1) 를 호가로 맞춘 값.
    positions = [row["position"] for row in view["rounds"]] + list(trades)
    view["position_prices"] = {
        position: ladder.round_to_tick(
            first["fill_price"] * ladder.down_ratio(trigger_pct) ** (position - 1),
            kind=view["instrument_kind"],
            country=view["country"],
        )
        for position in range(min(positions), max(positions) + 1)
    }


def _build_strategy_view(
    strategy_id: str,
    strategy_settings: dict[str, Any],
    account_closes: dict[str, float],
    ledger: Ledger,
    cash: float,
    *,
    kind: ladder.InstrumentKind,
    country: ladder.Country,
) -> dict[str, Any]:
    """전략 하나의 운용 현황(회차 표·상태)을 회차 원장에서 만든다."""
    pool = str(strategy_settings["pool"])
    ticker = str(strategy_settings["ticker"])
    name = str(strategy_settings["name"])
    trigger_pct = float(strategy_settings["trigger_pct"])
    round_size = int(strategy_settings["round_size"])
    max_rounds = int(strategy_settings["rounds"])
    exchange = strategy_settings["exchange"]
    if country == "us" and not exchange:
        raise ValueError("이 종목의 거래소가 저장돼 있지 않습니다 — 전략을 다시 저장하세요.")

    quantity = ledger.quantity
    regular_lots = [lot for lot in ledger.lots if lot.arbitrary_id is None]
    live_close = account_closes.get(ticker) or _load_unheld_closes([ticker], pool, country).get(ticker)
    index_status = _load_ticker_status(pool, ticker, name, trigger_pct, live_close, kind=kind, country=country)
    close = float(index_status["close"])
    # 정정을 합친 원주문별 매수 로트가 한 회차다. 설정 회차 수를 넘긴 수동 체결도 모두 표시한다.
    held_rounds = len(regular_lots)
    last_fill = regular_lots[-1].price if regular_lots else None
    close_entry = held_rounds == 0 and _first_entry_close_window(kind, country)

    # 팔 수 있는 회차는 맨 뒤(가장 깊은) 보유 회차 하나뿐이고, 신호는 한 번에 하나라
    # 그 매도가 닿아 있으면 매수는 매도가 체결될 때까지 미룬다.
    sell_pending = bool(
        held_rounds
        and close > 0
        and close >= ladder.sell_limit(regular_lots[-1].price, trigger_pct, kind=kind, country=country)
    )

    # 사다리 위치(표시용) — 보유 회차는 1호 체결가 기준 칸, 그 뒤 회차는 마지막 보유 칸부터 한 칸씩.
    held_positions = ladder.ladder_positions([lot.price for lot in regular_lots], trigger_pct)
    rows: list[dict[str, Any]] = []
    for round_no in range(1, max(held_rounds, max_rounds) + 1):
        held = round_no <= held_rounds
        position = (
            held_positions[round_no - 1]
            if held
            else (held_positions[-1] if held_positions else 0) + round_no - held_rounds
        )
        is_next = round_no == held_rounds + 1
        if held:
            fill = regular_lots[round_no - 1]
            sell_limit = ladder.sell_limit(fill.price, trigger_pct, kind=kind, country=country)
            # 이 회차를 팔면 다음 매수선은 앞 회차 체결가 −간격%다. 그게 매도가 이상이면 팔자마자 다시 사게 된다 —
            # 신호를 '매도 후 재매수' 한 짝으로 낸다.
            rebuy_after_sell = bool(
                round_no == held_rounds
                and round_no >= 2
                and ladder.projected_buy_limit(
                    regular_lots[round_no - 2].price, 1, trigger_pct, kind=kind, country=country
                )
                >= sell_limit
                and 0 < (rebuy_quantity := round_buy_quantity(round_size, sell_limit, country))
                and rebuy_quantity * sell_limit <= cash
            )
            rows.append(
                {
                    "round": round_no,
                    "fixed_round": round_no,
                    "position": position,
                    "ticker": ticker,
                    "name": name,
                    "held": True,
                    "quantity": fill.shares,
                    # 실제 체결가·산 날짜·출처(슬랙/수동).
                    "fill_price": ladder.round_price(fill.price, country),
                    "bought_on": fill.day,
                    "source": fill.source,
                    "close": ladder.round_price(close, country) if close > 0 else None,
                    "profit_pct": round((close / fill.price - 1.0) * 100.0, 2) if close > 0 else None,
                    "buy_limit": None,
                    "buy_index": ladder.round_price(
                        fill.price, country
                    ),  # 이미 산 회차 — 체결가가 그 회차의 기준선이다
                    "sell_limit": sell_limit,
                    "sell_index": sell_limit,
                    # 팔 수 있는 건 마지막 보유 회차뿐이다(앞 회차는 체결가가 더 높아 먼저 닿지도 않는다).
                    "sell_reached": bool(round_no == held_rounds and sell_pending),
                    "rebuy_after_sell": rebuy_after_sell,
                    "buy_reached": False,
                    "close_entry": False,
                    "cash_short": False,
                    "budget_short": False,
                    "is_next": False,
                }
            )
            continue
        # 미보유 회차 — 보유가 있으면 마지막 보유 체결가에서 내려간 자리(다음 회차 이후는 예상값),
        # 없으면 전일 종가 기준. 지정가로 바로 쓸 수 있게 호가 단위로 반올림한다.
        if last_fill is not None:
            buy_limit = ladder.projected_buy_limit(
                last_fill, round_no - held_rounds, trigger_pct, kind=kind, country=country
            )
        else:
            buy_limit = ladder.preview_buy_limit(
                float(index_status["prev_close"]), round_no, trigger_pct, kind=kind, country=country
            )
        market_entry = is_next and close_entry and close > buy_limit
        # 장 마감 매수는 현재가로 사므로 그 가격으로 산 금액을 본다.
        # 신호에 도달했다면 실제 시세로 예상하고, 대기 중에는 목표 지정가로 예상한다.
        buy_price = min(close, buy_limit) if is_next and 0 < close <= buy_limit else buy_limit
        if market_entry:
            buy_price = close
        buy_quantity = round_buy_quantity(round_size, buy_price, country)
        budget_short = is_next and buy_quantity == 0
        cash_short = is_next and buy_quantity * buy_price > cash
        rows.append(
            {
                "round": round_no,
                "fixed_round": round_no,
                "position": position,
                "ticker": ticker,
                "name": name,
                "held": False,
                "quantity": buy_quantity,
                "fill_price": None,
                "bought_on": None,
                "source": None,
                "close": ladder.round_price(close, country) if close > 0 else None,
                "profit_pct": None,
                "buy_limit": buy_limit,
                "buy_index": buy_limit,
                "sell_limit": None,
                "sell_index": None,
                "sell_reached": False,
                "rebuy_after_sell": False,
                "buy_reached": bool(
                    is_next
                    and not sell_pending
                    and not cash_short
                    and not budget_short
                    and close > 0
                    and (close <= buy_limit or close_entry)
                ),
                # 진입선이 아니라 장 마감 매수로 닿은 1호 — 지정가 대신 현재가로 산다.
                "close_entry": bool(market_entry and not cash_short and not budget_short),
                # 다음 회차를 살 현금이 없다 — 한도 소진, 이 회차는 사지 않는다.
                "cash_short": cash_short,
                "budget_short": budget_short,
                "is_next": is_next,
            }
        )

    for row in rows:
        row["manual_orders"] = [
            order
            for order in ledger.manual_orders
            if (order["side"] == "buy" and row["is_next"]) or (order["side"] == "sell" and row["round"] == held_rounds)
        ]

    # 다음 회차 매수까지 남은 하락률(%) — 도달했으면 0 이상(+). 회차 소진이면 None.
    next_buy_drop_pct = None
    next_row = next((r for r in rows if r["is_next"]), None)
    if next_row is not None and next_row["buy_limit"] is not None and close > 0:
        next_buy_drop_pct = round((next_row["buy_limit"] / close - 1.0) * 100.0, 2)

    # 가장 가까운 매도 = 마지막(가장 싼) 회차 — 체결가가 가장 낮아 목표도 가장 먼저 닿는다.
    next_sell_round = None
    next_sell_rise_pct = None
    held_rows = [r for r in rows if r["held"] and r["sell_limit"] is not None and close > 0]
    if held_rows:
        deepest = held_rows[-1]
        next_sell_round = deepest["round"]
        next_sell_rise_pct = round((deepest["sell_limit"] / close - 1.0) * 100.0, 2)

    return {
        "strategy_id": strategy_id,
        "label": name,
        "pool": pool,
        "instrument_kind": kind,
        "country": country,
        "exchange": exchange,
        "ticker": ticker,
        "config": {
            "trigger_pct": trigger_pct,
            "round_size": round_size,
            "rounds": max_rounds,
            "index_name": name,
        },
        "index": index_status,
        "cash": cash,
        "status": {
            "held_count": held_rounds,
            "waiting_first_entry": held_rounds == 0,
            "first_entry_close_time": FIRST_ENTRY_CLOSE_TIMES[country][kind].strftime("%H:%M"),
            "next_round": (held_rounds + 1) if held_rounds < max_rounds else None,
            "next_ticker": ticker if held_rounds < max_rounds else None,
            "next_name": name if held_rounds < max_rounds else None,
            "last_buy_price": ladder.round_price(last_fill, country) if last_fill is not None else None,
            "next_buy_drop_pct": next_buy_drop_pct,
            "next_sell_round": next_sell_round,
            "next_sell_rise_pct": next_sell_rise_pct,
            # 매도가 먼저 닿아 있어 다음 회차 매수는 그 매도가 체결될 때까지 미룬다(신호는 한 번에 하나).
            "sell_pending": sell_pending,
            "held_quantity": quantity,
            # 다음 회차를 살 현금이 없어 매수를 멈췄는지(한도 소진).
            "cash_short": bool(next_row is not None and next_row["cash_short"]),
            "budget_short": bool(next_row is not None and next_row["budget_short"]),
        },
        # 증권사에 걸린 우리 기록에 없는(수동) 미체결 주문 — 슬랙 주문을 막는 근거다.
        "manual_orders": list(ledger.manual_orders),
        # 이 현황의 체결 이력을 계산한 시각 — 화면이 기준 시점을 보여 준다.
        "ledger_at": ledger.computed_at,
        "arbitrary_lots": [
            {
                "id": lot.arbitrary_id,
                "quantity": lot.shares,
                "fill_price": lot.price,
                "bought_on": lot.day,
                "sell_limit": ladder.sell_limit(lot.price, trigger_pct, kind=kind, country=country),
            }
            for lot in ledger.lots
            if lot.arbitrary_id is not None
        ],
        "rounds": rows,
    }


def _tickers_by_account(strategies: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """전략 종목을 운용 계좌(계좌 ID)별로 모은다 — 증권사 조회는 계좌당 한 벌이다."""
    grouped: dict[str, set[str]] = {}
    for entry in strategies.values():
        grouped.setdefault(entry["account_id"], set()).add(entry["ticker"])
    return {account_id: sorted(tickers) for account_id, tickers in grouped.items()}


def refresh_all_ledgers() -> dict[str, str]:
    """모든 전략의 회차 원장을 증권사에서 새로 계산해 저장한다 — 사고팔기 알림 배치가 돈다.

    실패한 것은 (``계좌ID:티커`` → 이유)다. 계좌 하나의 연동 오류가 다른 계좌를 막지 않는다.
    """
    from services.broker_api_service import BrokerApiError

    strategies = load_settings()["strategies"]
    failed: dict[str, str] = {}
    for account_id, tickers in _tickers_by_account(strategies).items():
        try:
            ledgers = refresh_ledgers(account_id, tickers)
        except BrokerApiError as error:
            failed.update({f"{account_id}:{ticker}": str(error) for ticker in tickers})
            continue
        for ticker, ledger in ledgers.items():
            if isinstance(ledger, LedgerError):
                failed[f"{account_id}:{ticker}"] = str(ledger)
        forget_ledgers(account_id, tickers)
    return failed


def _failed_view(strategy_id: str, entry: dict[str, Any], error: LedgerError | str) -> dict[str, Any]:
    """운용 현황을 만들 수 없는 전략 — 회차 표 없이 이유만 담는다(다른 전략은 그대로 본다)."""
    return {
        "strategy_id": strategy_id,
        "label": str(entry["name"]),
        "pool": str(entry["pool"]),
        "ticker": str(entry["ticker"]),
        "config": {
            "trigger_pct": entry["trigger_pct"],
            "round_size": entry["round_size"],
            "rounds": entry["rounds"],
            "index_name": str(entry["name"]),
        },
        "error": str(error),
        "rounds": [],
    }


def _account_ledgers(
    tickers_by_account: dict[str, list[str]], *, fresh: bool
) -> dict[str, dict[str, Ledger | LedgerError] | str]:
    """계좌별 회차 원장 — 증권사 연동이 없거나 조회에 실패한 계좌는 원장 대신 이유(문자열)다(그 계좌 전략만 실패로 본다).

    저장본 읽기(``fresh`` 거짓)는 증권사를 부르지 않지만, 연동이 없는 계좌는 같은 이유로 실패로 보여 준다.
    """
    from services.broker_api_service import BrokerApiError
    from services.broker_trade_service import require_trade_account

    load = refresh_ledgers if fresh else read_ledgers
    ledgers: dict[str, dict[str, Ledger | LedgerError] | str] = {}
    for account_id, tickers in tickers_by_account.items():
        try:
            require_trade_account(account_id)
            ledgers[account_id] = load(account_id, tickers)
        except BrokerApiError as error:
            ledgers[account_id] = str(error)
    return ledgers


def _build_views(
    strategies: dict[str, dict[str, Any]], *, fresh: bool, include_trade_history: bool
) -> list[dict[str, Any]]:
    """전략들의 운용 현황 — 전략마다 고른 운용 계좌의 회차 원장으로 만든다.

    ``fresh`` 가 참이면 원장을 증권사에서 새로 계산해 저장하고(느리다), 아니면 배치가 저장해 둔 것을 읽는다.

    증권사 연동이 없거나 회차 원장을 못 만든 전략은 `error` 만 담은 항목으로 돌려준다(`_failed_view`)
    — 한 전략의 문제가 다른 전략을 막지 않는다.
    """
    tickers_by_account = _tickers_by_account(strategies)
    ledgers = _account_ledgers(tickers_by_account, fresh=fresh)
    pool_countries = _pool_countries(TRADE_COUNTRIES)
    account_countries = {
        entry["account_id"]: pool_countries[entry["pool"]]
        for entry in strategies.values()
        if entry["pool"] in pool_countries
    }
    states = {
        account_id: _load_account_state(account_id, account_countries[account_id])
        for account_id, account_ledgers in ledgers.items()
        if not isinstance(account_ledgers, str) and account_id in account_countries
    }
    from utils.account_settings_store import load_account_docs
    from utils.strategy_trade_orders import read_trade_history_orders

    names = {doc["account_id"]: str(doc.get("name") or doc["account_id"]) for doc in load_account_docs()}
    active_pools = _eligible_pool_kinds(TRADE_COUNTRIES) if strategies else {}
    source_tickers = {
        pool: _pool_tickers(pool, active_pools[pool])
        for pool in {entry["pool"] for entry in strategies.values()} & active_pools.keys()
    }
    views: list[dict[str, Any]] = []
    cached_days_by_account: dict[tuple[str, date], dict[date, list[dict[str, Any]]]] = {}
    history_orders_by_account: dict[tuple[str, date], list[dict[str, Any]]] = {}
    for sid, entry in strategies.items():
        account_id = entry["account_id"]
        account_ledgers = ledgers[account_id]
        if entry["ticker"] not in source_tickers.get(entry["pool"], {}):
            view = _failed_view(sid, entry, f"가격 출처 '{entry['pool']}'에 종목이 없습니다. 종목을 다시 선택하세요.")
        elif isinstance(account_ledgers, str):
            view = _failed_view(sid, entry, account_ledgers)
        elif isinstance(ledger := account_ledgers[entry["ticker"]], LedgerError):
            view = _failed_view(sid, entry, ledger)
        else:
            state = states[account_id]
            try:
                view = _build_strategy_view(
                    sid,
                    entry,
                    state["closes"],
                    ledger,
                    state["cash"],
                    kind=active_pools[entry["pool"]],
                    country=pool_countries[entry["pool"]],
                )
                if include_trade_history:
                    cache_key = (account_id, ledger.as_of)
                    if cache_key not in cached_days_by_account:
                        cached_days_by_account[cache_key] = read_cached_trade_days(*cache_key)
                        history_orders_by_account[cache_key] = read_trade_history_orders(*cache_key)
                    orders = [
                        order for order in history_orders_by_account[cache_key] if order["ticker"] == view["ticker"]
                    ]
                    _attach_trade_history(view, ledger, cached_days_by_account[cache_key], orders)
            except (RuntimeError, ValueError) as error:
                view = _failed_view(sid, entry, str(error))
        # 운용 계좌 — 화면 카드 머리·슬랙 문구·주문 계좌가 이것을 쓴다.
        views.append({**view, "account_id": account_id, "account_name": names.get(account_id, account_id)})
    return views


def load_views_where(keep: Callable[[dict[str, Any]], bool], *, fresh: bool) -> dict[str, Any]:
    """``keep(전략 설정)`` 이 참인 전략들의 운용 현황 — 알림·주문용. ``fresh`` 는 `_build_views` 와 같다.

    회차 원장을 못 만든 전략은 `strategies` 에서 빼고 `errors`(전략 id → 이유)에 담는다 — 그 전략만 건너뛴다.
    """
    views = _build_views(
        {sid: entry for sid, entry in load_settings()["strategies"].items() if keep(entry)},
        fresh=fresh,
        include_trade_history=False,
    )
    return {
        "strategies": [view for view in views if "error" not in view],
        "errors": {view["strategy_id"]: view["error"] for view in views if "error" in view},
    }


def strategy_entry(strategy_id: str) -> dict[str, Any]:
    """저장된 전략 설정 하나 — 없으면 에러."""
    entry = load_settings()["strategies"].get(strategy_id)
    if entry is None:
        raise ValueError(f"'{strategy_id}' 사고팔기 전략이 없습니다.")
    return entry


def _trade_account_docs() -> list[dict[str, Any]]:
    """증권사 API가 연동된 운영 계좌(국가는 `TRADE_COUNTRIES`)."""
    from utils.account_settings_store import load_live_account_docs

    return [
        doc
        for doc in load_live_account_docs()
        if (doc.get("broker_api") or {}).get("provider")
        and str(doc.get("country_code") or "").strip().lower() in TRADE_COUNTRIES
    ]


def _usd_krw() -> float | None:
    """미장 화면의 원화금액 컬럼용 달러→원 환율 — 못 받으면 None 으로 두어 컬럼을 빈 값으로 보인다."""
    from services.price_service import get_exchange_rates

    try:
        return float(get_exchange_rates(["USD"])["USD"]["rate"])
    except Exception:
        logger.warning("사고팔기 달러 환율 조회 실패 — 원화금액을 표시하지 않습니다", exc_info=True)
        return None


def load_strategy_trade_view(account_id: str | None) -> dict[str, Any]:
    """화면용 — 사고팔기 계좌 목록과, 고른 계좌의 슬랙 스위치·전략 설정·운용 현황.

    ``account_id`` 가 None 이면 첫 계좌를 보여준다. 운영할 수 있는 계좌가 없으면 ``account_id`` 는 None 이다.
    """
    docs = _trade_account_docs()
    if account_id is not None and account_id not in {doc["account_id"] for doc in docs}:
        raise ValueError(f"'{account_id}' 는 사고팔기를 운영할 수 있는 계좌가 아닙니다(증권사 API 연동 필요).")
    selected = account_id if account_id is not None else (docs[0]["account_id"] if docs else None)
    country = next((str(doc["country_code"]).strip().lower() for doc in docs if doc["account_id"] == selected), None)
    settings = load_settings()
    account = settings["accounts"].get(selected or "", {"slack_enabled": False})
    mine = {sid: entry for sid, entry in settings["strategies"].items() if entry["account_id"] == selected}
    # 순서를 저장한 적 없는 전략끼리는 기존 문서 순서를 유지한다.
    configs = sorted(mine.items(), key=lambda item: (item[1]["sort_order"] is None, item[1]["sort_order"]))
    from utils.pool_strategy_use import pool_options

    pools = _eligible_pool_kinds((country,)) if selected is not None else {}
    options = _ticker_options(pools)
    return {
        "account_options": [
            {
                "account_id": item["account_id"],
                "name": item.get("name"),
                "icon": item.get("icon"),
                "order": item.get("order"),
                "country_code": str(item["country_code"]).strip().lower(),
            }
            for item in docs
        ],
        "account_id": selected,
        "country_code": country,
        "slack_enabled": account["slack_enabled"],
        "groups": _normalize_trade_groups(account.get("groups", [])),
        "pool_options": pool_options(list(pools)),
        "ticker_options": [options[ticker] for ticker in sorted(options)],
        "default_round_size": DEFAULT_ROUND_SIZES[country] if country else None,
        "usd_krw": _usd_krw() if country == "us" else None,
        # 편집 표의 행 — 저장된 전략 설정 그대로.
        "configs": [
            {
                "strategy_id": sid,
                "pool": entry["pool"],
                "ticker": entry["ticker"],
                "name": entry["name"],
                "trigger_pct": entry["trigger_pct"],
                "round_size": entry["round_size"],
                "rounds": entry["rounds"],
            }
            for sid, entry in configs
        ],
        "strategies": _build_views(mine, fresh=False, include_trade_history=True),
    }
