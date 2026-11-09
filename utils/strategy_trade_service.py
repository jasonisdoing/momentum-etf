"""전략 사고팔기 운용 현황 조회 서비스 (UI/API 용).

전략 여러 개를 독립 운용한다. 각 전략은 **자기 종목 하나만** 설정된 회차 수까지 분할 매수한다.
여러 한국 개별주·ETF 종목풀에서 전략 종목을 고르며, 전략별 종목풀은 가격 출처다.
파라미터(매매 간격 %·회차당 수량·회차 수)는
DB(``system_config.strategy_trade_settings.strategies``)가 단일 소스이며 화면에서
편집한다. 슬랙 알람은 계좌별 사고팔기 스위치(``accounts.{계좌}.slack_enabled``)를 따른다 — 합성과 별개다.

사고팔기는 합성과 별개로 증권사 API가 연동된 한국 운영 계좌에 전략을 직접 등록한다. 다음 회차를 계좌 원화 예수금(잔고 동기화 값)으로 살 수 없으면 한도 소진으로 사지 않는다.
보유 현황은 전략의 운용 계좌의 **증권사 체결 이력**에서 만든다(백테스트는 없다).
화면을 열 때마다 `strategy_trade_ledger` 가 체결을 재생해 회차별 실제 체결가를 구한다(슬랙 신호 주문·수동 주문 모두).

사다리 규칙 (자세한 근거는 strategy_trade_config 모듈 설명 참고)
---------------------------------------------------------------
- 진입 회차 수 m = 보유 수량 ÷ 회차당 수량 (반올림, 1~회차 수로 제한).
- 1호가 P1 = 1호를 실제로 산 체결가,  r = 1 − 간격%.
- k호 진입가 = P1 × r^(k-1) · 매도 지정가 = 진입가 × (1 + 간격%). 갭으로 싸게 산 회차도 선은 그대로다.
- 다음 매수 = 마지막 매수가 × r. 보유가 없으면 1호 진입 = 전일 종가 × r,
  장중에 못 닿았으면 15:00(``FIRST_ENTRY_CLOSE_TIME``)부터 마감 전까지 현재가로 1호 매수.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from core.strategy.trade import rules as ladder
from utils.strategy_trade_config import (
    EDITABLE_CONFIG_KEYS,
    FIRST_ENTRY_CLOSE_TIME,
    validate_strategy_trade_config,
)
from utils.strategy_trade_ledger import (
    Ledger,
    LedgerError,
    forget_ledgers,
    read_cached_trade_days,
    read_ledgers,
    refresh_ledgers,
    round_history,
)

KST = ZoneInfo("Asia/Seoul")
_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def _settings_collection():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 전략 사고팔기 설정을 읽을 수 없습니다.")
    return db[_CONFIG_COLLECTION]


def load_settings() -> dict[str, Any]:
    """계좌별 슬랙 설정과 전략별 가격 출처·파라미터를 반환한다."""
    doc = _settings_collection().find_one({"_id": _SETTINGS_KEY}) or {}
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
        accounts[str(account_id)] = {"slack_enabled": entry["slack_enabled"]}

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
                **validate_strategy_trade_config({key: entry.get(key) for key in EDITABLE_CONFIG_KEYS}),
            }
        except ValueError as error:
            raise ValueError(f"'{sid}' 전략 설정이 올바르지 않습니다: {error}") from error

    return {"accounts": accounts, "strategies": strategies}


def _eligible_pool_kinds() -> dict[str, ladder.InstrumentKind]:
    """활성 국내 개별주·ETF 풀과 호가 계산에 사용할 종목 유형."""
    from utils.settings_loader import _load_pool_configs
    from utils.strategy_trade_config import is_eligible_pool

    return {
        str(config["ticker_type"]): "stock" if str(config["pool_kind"]).strip().lower() == "stock" else "etf"
        for config in _load_pool_configs()
        if is_eligible_pool(config)
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
    account_id: str, config: dict[str, Any], current: dict[str, Any] | None, options: dict[str, dict[str, Any]]
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
            raise ValueError(f"사고팔기에 등록할 수 있는 한국 개별주·ETF가 아닙니다: {ticker}")
        candidates = option["pools"]
        # 기존 출처가 여전히 유효하면 같은 종목의 기준가를 바꾸지 않는다.
        pool = (
            current["pool"]
            if current and ticker == current["ticker"] and current["pool"] in candidates
            else candidates[0]
        )
        name = option["name"]
    return {
        "pool": pool,
        "account_id": account_id,
        "ticker": ticker,
        "name": name,
        **validate_strategy_trade_config(config),
    }


def _sent_log_unset(sent_log: dict[str, Any], strategy_id: str) -> dict[str, str]:
    """그 전략의 알림 중복 방지 기록을 지우는 $unset 항목."""
    return {f"sent_log.{key}": "" for key in sent_log if str(key).startswith(f"{strategy_id}:")}


def save_account_trade(account_id: str, slack_enabled: Any, strategies: Any) -> None:
    """계좌의 슬랙 설정과 전략 목록을 검증한 뒤 한 번에 저장한다."""
    from uuid import uuid4

    if not any(item["account_id"] == account_id for item in _trade_account_docs()):
        raise ValueError(f"'{account_id}' 는 사고팔기를 운영할 수 있는 계좌가 아닙니다(증권사 API 연동 필요).")
    if not isinstance(slack_enabled, bool):
        raise ValueError("slack_enabled 는 참·거짓이어야 합니다.")
    if not isinstance(strategies, list):
        raise ValueError("strategies 형식이 올바르지 않습니다.")
    options = _ticker_options(_eligible_pool_kinds())
    current = {sid: entry for sid, entry in load_settings()["strategies"].items() if entry["account_id"] == account_id}
    taken: set[str] = set()
    seen_ids: set[str] = set()
    planned: dict[str, dict[str, Any]] = {}
    for row in strategies:
        if not isinstance(row, dict):
            raise ValueError("전략 행의 형식이 올바르지 않습니다.")
        sid = str(row.get("strategy_id") or "").strip()
        if sid and (sid not in current or sid in seen_ids):
            raise ValueError(f"이 계좌의 전략이 아니거나 중복된 전략입니다: {sid}")
        if sid:
            seen_ids.add(sid)
        previous = current.get(sid)
        target = _validated_target(account_id, row, previous, options)
        if target["ticker"] in taken:
            raise ValueError(f"{target['name']}({target['ticker']}) 전략이 이 계좌에 이미 있습니다.")
        taken.add(target["ticker"])
        # 종목이 바뀌면 이전 주문 버튼이 새 종목을 가리키지 않도록 별도 전략으로 저장한다.
        target_id = sid if previous and previous["ticker"] == target["ticker"] else f"s{uuid4().hex[:8]}"
        planned[target_id] = target

    collection = _settings_collection()
    stored = collection.find_one({"_id": _SETTINGS_KEY}, {"sent_log": 1, "accounts": 1}) or {}
    sent_log = stored.get("sent_log") or {}
    account_settings: dict[str, Any] = {"slack_enabled": slack_enabled}
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


def _load_account_state(account_id: str) -> dict[str, Any]:
    """계좌 보유 종목의 현재가와 현금 (전 전략 공용, 1회 조회).

    현재가는 ``load_real_holdings_table`` 이 실시간 시세로 채운 값이다. 수량·평균단가는 쓰지 않는다 —
    보유는 증권사 체결 이력(회차 원장)이 정한다. 현금은 잔고 동기화가 저장한 값(자산 화면과 같은 값)이다 —
    ``cash_krw`` 는 매수에 쓰는 원화 예수금, ``total_krw`` 는 보유 평가액 + 현금(원화 환산) 계좌 평가액이다.
    """
    from utils.broker_balance_sync import stored_cash_map
    from utils.portfolio_io import load_portfolio_master, load_real_holdings_table

    master = load_portfolio_master(account_id)
    if master is None:
        raise ValueError(f"'{account_id}' 계좌의 잔고가 저장돼 있지 않습니다 — 잔고 동기화를 먼저 돌리세요.")
    cash_krw = stored_cash_map(master, ["KRW"])["KRW"]
    frame = load_real_holdings_table(account_id)
    closes: dict[str, float] = {}
    holdings_value = 0.0
    if frame is not None and not frame.empty:
        missing = {"티커", "현재가", "평가금액(KRW)"} - set(frame.columns)
        if missing:
            raise RuntimeError(f"보유 테이블에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")
        for _, row in frame.iterrows():
            ticker = str(row["티커"]).strip()
            if ticker:
                closes[ticker] = float(row["현재가"] or 0)
                holdings_value += float(row["평가금액(KRW)"] or 0)
    return {
        "closes": closes,
        "cash_krw": cash_krw,
        "total_krw": holdings_value + float(master.get("cash_balance") or 0),
    }


def _load_unheld_closes(tickers: list[str], pool: str) -> dict[str, float]:
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

        snapshot = get_realtime_snapshot("kor", tickers)
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
RECENT_LOW_MONTHS = 12


def _load_ticker_status(
    pool: str, ticker: str, name: str, trigger_pct: float, live_close: float | None, *, kind: ladder.InstrumentKind
) -> dict[str, Any]:
    """판정 기준 시세 — 이 전략 종목 자신의 현재가·전일 종가·1호 진입선·최근 저점.

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

    # 최근 RECENT_LOW_MONTHS 개월 저점 — 회차 지정가가 어느 층에 걸리는지 가늠하는 기준선.
    # 가격 캐시는 확정 종가라 오늘 장중 현재가가 새 저점이어도 빠져 있다 — 현재가가 더 낮으면
    # 오늘이 저점이다(그대로 두면 현재가가 기간 저점보다 낮은 값으로 표시됐다).
    low_window = close[close.index >= close.index[-1] - pd.DateOffset(months=RECENT_LOW_MONTHS)]
    low_value, low_date = float(low_window.min()), low_window.idxmin()
    if current < low_value:
        low_value, low_date = current, today

    return {
        "name": name,
        "as_of": close.index[-1].strftime("%Y-%m-%d"),
        "close": round(current),
        "prev_close": round(prev_close),
        "change_pct": round(change_pct, 2),
        # 1호 진입선 — 전일 종가 대비 -간격%, 호가 단위 반올림.
        "buy_trigger": ladder.first_buy_limit(prev_close, float(trigger_pct), kind=kind),
        "recent_low": round(low_value),
        "recent_low_date": low_date.strftime("%Y-%m-%d"),
        "recent_low_label": f"{RECENT_LOW_MONTHS}개월 저점",
    }


def _first_entry_close_window(now: datetime | None = None) -> bool:
    """지금이 1호 장 마감 매수 시간인지 — 한국 거래일의 ``FIRST_ENTRY_CLOSE_TIME`` ~ 정규장 마감 전."""
    from zoneinfo import ZoneInfo

    from config import MARKET_SCHEDULES
    from utils.market_session import trading_day_today

    schedule = MARKET_SCHEDULES["kor"]
    now_local = (now or datetime.now(ZoneInfo(schedule["timezone"]))).astimezone(ZoneInfo(schedule["timezone"]))
    if trading_day_today("kor", now_local) is None:
        return False
    return FIRST_ENTRY_CLOSE_TIME <= now_local.time() < schedule["close"]


def _attach_trade_history(view: dict[str, Any], ledger: Ledger, cached_days: dict[date, list[dict[str, Any]]]) -> None:
    """회차 행에 날짜별 매수·매도를 체결 순서대로 붙인다(`trades_by_day` = {날짜: ["buy", "sell", …]}).

    지금 사이클의 체결만 담는다 — 1호를 마지막으로 판 체결(그때 보유가 모두 비었다) 뒤부터다.
    그 전 사이클의 매매는 지금 보유와 무관해 보여 주지 않는다.
    """
    rows = {row["round"]: row for row in view["rounds"]}
    for row in rows.values():
        row["trades_by_day"] = {}
    events = round_history(ledger, view["ticker"], view["config"]["round_quantity"], cached_days)
    start = next(
        (
            idx + 1
            for idx in range(len(events) - 1, -1, -1)
            if events[idx]["round"] == 1 and events[idx]["side"] == "sell"
        ),
        0,
    )
    for event in events[start:]:
        if (row := rows.get(event["round"])) is not None:
            row["trades_by_day"].setdefault(event["day"], []).append(event["side"])


def _build_strategy_view(
    strategy_id: str,
    strategy_settings: dict[str, Any],
    account_closes: dict[str, float],
    ledger: Ledger,
    cash_krw: float,
    *,
    kind: ladder.InstrumentKind,
) -> dict[str, Any]:
    """전략 하나의 운용 현황(회차 표·상태)을 회차 원장에서 만든다.

    ``cash_krw`` 는 계좌의 원화 예수금 — 다음 회차 금액(회차당 수량 × 매수가)을 살 수 없으면 한도 소진으로 사지 않는다.
    """
    pool = str(strategy_settings["pool"])
    ticker = str(strategy_settings["ticker"])
    name = str(strategy_settings["name"])
    trigger_pct = float(strategy_settings["trigger_pct"])
    round_quantity = int(strategy_settings["round_quantity"])
    max_rounds = int(strategy_settings["rounds"])

    quantity = ledger.quantity
    live_close = account_closes.get(ticker) or _load_unheld_closes([ticker], pool).get(ticker)
    index_status = _load_ticker_status(pool, ticker, name, trigger_pct, live_close, kind=kind)
    close = float(index_status["close"])
    # 보유 0 인데 장중 1호 진입선에 못 닿은 채 장 마감이 다가오면 현재가로 1호를 산다.
    close_entry = False

    # ── 회차 — 수량으로 진입 회차 수, 마지막 보유 회차의 체결가로 다음 매수가 ──
    held_rounds = 0
    last_fill: float | None = None
    quantity_mismatch = False
    round_fills: list[dict[str, Any]] = []
    if quantity > 0:
        held_rounds = max(1, min(max_rounds, round(quantity / round_quantity)))
        quantity_mismatch = quantity != held_rounds * round_quantity
        round_fills = ledger.round_fills(round_quantity)
        last_fill = round_fills[held_rounds - 1]["price"]
    else:
        close_entry = _first_entry_close_window()

    # 팔 수 있는 회차는 맨 뒤(가장 깊은) 보유 회차 하나뿐이고, 신호는 한 번에 하나라
    # 그 매도가 닿아 있으면 매수는 매도가 체결될 때까지 미룬다.
    sell_pending = bool(
        held_rounds
        and close > 0
        and close >= ladder.sell_limit(round_fills[held_rounds - 1]["price"], trigger_pct, kind=kind)
    )

    # 사다리 위치(표시용) — 보유 회차는 1호 체결가 기준 칸, 그 뒤 회차는 마지막 보유 칸부터 한 칸씩.
    held_positions = ladder.ladder_positions([fill["price"] for fill in round_fills[:held_rounds]], trigger_pct)
    rows: list[dict[str, Any]] = []
    for round_no in range(1, max_rounds + 1):
        held = round_no <= held_rounds
        position = (
            held_positions[round_no - 1]
            if held
            else (held_positions[-1] if held_positions else 0) + round_no - held_rounds
        )
        is_next = round_no == held_rounds + 1
        if held:
            fill = round_fills[round_no - 1]
            sell_limit = ladder.sell_limit(fill["price"], trigger_pct, kind=kind)
            rows.append(
                {
                    "round": round_no,
                    "fixed_round": round_no,
                    "position": position,
                    "ticker": ticker,
                    "name": name,
                    "held": True,
                    # 실제 체결가·산 날짜·출처(슬랙/수동).
                    "fill_price": round(fill["price"]),
                    "bought_on": fill["day"],
                    "source": fill["source"],
                    "close": round(close) if close > 0 else None,
                    "profit_pct": round((close / fill["price"] - 1.0) * 100.0, 2) if close > 0 else None,
                    "buy_limit": None,
                    "buy_index": round(fill["price"]),  # 이미 산 회차 — 체결가가 그 회차의 기준선이다
                    "sell_limit": sell_limit,
                    "sell_index": sell_limit,
                    # 팔 수 있는 건 마지막 보유 회차뿐이다(앞 회차는 체결가가 더 높아 먼저 닿지도 않는다).
                    "sell_reached": bool(round_no == held_rounds and sell_pending),
                    "buy_reached": False,
                    "close_entry": False,
                    "cash_short": False,
                    "is_next": False,
                }
            )
            continue
        # 미보유 회차 — 보유가 있으면 마지막 보유 체결가에서 내려간 자리(다음 회차 이후는 예상값),
        # 없으면 전일 종가 기준. 지정가로 바로 쓸 수 있게 호가 단위로 반올림한다.
        if last_fill is not None:
            buy_limit = ladder.projected_buy_limit(last_fill, round_no - held_rounds, trigger_pct, kind=kind)
        else:
            buy_limit = ladder.preview_buy_limit(float(index_status["prev_close"]), round_no, trigger_pct, kind=kind)
        market_entry = is_next and close_entry and close > buy_limit
        # 15:00 시장가 매수는 현재가로 사므로 그 가격으로 산 금액을 본다.
        cash_short = bool(is_next and round_quantity * (close if market_entry else buy_limit) > cash_krw)
        rows.append(
            {
                "round": round_no,
                "fixed_round": round_no,
                "position": position,
                "ticker": ticker,
                "name": name,
                "held": False,
                "fill_price": None,
                "bought_on": None,
                "source": None,
                "close": round(close) if close > 0 else None,
                "profit_pct": None,
                "buy_limit": buy_limit,
                "buy_index": buy_limit,
                "sell_limit": None,
                "sell_index": None,
                "sell_reached": False,
                "buy_reached": bool(
                    is_next
                    and not sell_pending
                    and not cash_short
                    and close > 0
                    and (close <= buy_limit or close_entry)
                ),
                # 진입선이 아니라 장 마감 매수로 닿은 1호 — 지정가 대신 현재가로 산다.
                "close_entry": bool(market_entry and not cash_short),
                # 다음 회차를 살 현금이 없다 — 한도 소진, 이 회차는 사지 않는다.
                "cash_short": cash_short,
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

    last_buy_price = round(round_fills[held_rounds - 1]["price"]) if held_rounds else None
    return {
        "strategy_id": strategy_id,
        "label": name,
        "pool": pool,
        "instrument_kind": kind,
        "ticker": ticker,
        "config": {
            "trigger_pct": trigger_pct,
            "round_quantity": round_quantity,
            "rounds": max_rounds,
            "index_name": name,
        },
        "index": index_status,
        "status": {
            "held_count": held_rounds,
            "waiting_first_entry": held_rounds == 0,
            "first_entry_close_time": FIRST_ENTRY_CLOSE_TIME.strftime("%H:%M"),
            "next_round": (held_rounds + 1) if held_rounds < max_rounds else None,
            "next_ticker": ticker if held_rounds < max_rounds else None,
            "next_name": name if held_rounds < max_rounds else None,
            "last_buy_price": last_buy_price,
            "next_buy_drop_pct": next_buy_drop_pct,
            "next_sell_round": next_sell_round,
            "next_sell_rise_pct": next_sell_rise_pct,
            # 매도가 먼저 닿아 있어 다음 회차 매수는 그 매도가 체결될 때까지 미룬다(신호는 한 번에 하나).
            "sell_pending": sell_pending,
            # 보유 수량이 회차×수량과 다르면 마지막 회차가 부분 보유다 — 화면이 경고를 띄운다.
            "quantity_mismatch": quantity_mismatch,
            "held_quantity": quantity,
            # 다음 회차를 살 현금이 없어 매수를 멈췄는지(한도 소진).
            "cash_short": bool(next_row is not None and next_row["cash_short"]),
        },
        # 증권사에 걸린 우리 기록에 없는(수동) 미체결 주문 — 슬랙 주문을 막는 근거다.
        "manual_orders": list(ledger.manual_orders),
        # 이 현황의 체결 이력을 계산한 시각 — 화면이 기준 시점을 보여 준다.
        "ledger_at": ledger.computed_at,
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
            "round_quantity": entry["round_quantity"],
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
    states = {
        account_id: _load_account_state(account_id)
        for account_id, account_ledgers in ledgers.items()
        if not isinstance(account_ledgers, str)
    }
    from utils.account_settings_store import load_account_docs

    names = {doc["account_id"]: str(doc.get("name") or doc["account_id"]) for doc in load_account_docs()}
    active_pools = _eligible_pool_kinds() if strategies else {}
    source_tickers = {
        pool: _pool_tickers(pool, active_pools[pool])
        for pool in {entry["pool"] for entry in strategies.values()} & active_pools.keys()
    }
    views: list[dict[str, Any]] = []
    cached_days_by_account: dict[tuple[str, date], dict[date, list[dict[str, Any]]]] = {}
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
                    sid, entry, state["closes"], ledger, state["cash_krw"], kind=active_pools[entry["pool"]]
                )
                if include_trade_history:
                    cache_key = (account_id, ledger.as_of)
                    if cache_key not in cached_days_by_account:
                        cached_days_by_account[cache_key] = read_cached_trade_days(*cache_key)
                    _attach_trade_history(view, ledger, cached_days_by_account[cache_key])
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
    """증권사 API가 연동된 한국 운영 계좌."""
    from utils.account_settings_store import load_live_account_docs

    return [
        doc
        for doc in load_live_account_docs()
        if (doc.get("broker_api") or {}).get("provider") and str(doc.get("country_code") or "").strip().lower() == "kor"
    ]


def load_strategy_trade_view(account_id: str | None) -> dict[str, Any]:
    """화면용 — 사고팔기 계좌 목록과, 고른 계좌의 슬랙 스위치·전략 설정·운용 현황.

    ``account_id`` 가 None 이면 첫 계좌를 보여준다. 운영할 수 있는 계좌가 없으면 ``account_id`` 는 None 이다.
    """
    docs = _trade_account_docs()
    if account_id is not None and account_id not in {doc["account_id"] for doc in docs}:
        raise ValueError(f"'{account_id}' 는 사고팔기를 운영할 수 있는 계좌가 아닙니다(증권사 API 연동 필요).")
    selected = account_id if account_id is not None else (docs[0]["account_id"] if docs else None)
    settings = load_settings()
    account = settings["accounts"].get(selected or "", {"slack_enabled": False})
    mine = {sid: entry for sid, entry in settings["strategies"].items() if entry["account_id"] == selected}
    from utils.pool_strategy_use import pool_options

    pools = _eligible_pool_kinds() if selected is not None else {}
    options = _ticker_options(pools)
    return {
        "account_options": [
            {
                "account_id": item["account_id"],
                "name": item.get("name"),
                "icon": item.get("icon"),
                "order": item.get("order"),
            }
            for item in docs
        ],
        "account_id": selected,
        "slack_enabled": account["slack_enabled"],
        # 계좌 현금(잔고 동기화 값)과 계좌 평가액 대비 비율 — 다음 회차를 살 수 없으면 한도 소진이다.
        "cash": _cash_summary(selected) if selected is not None else None,
        "pool_options": pool_options(list(pools)),
        "ticker_options": [options[ticker] for ticker in sorted(options)],
        # 편집 표의 행 — 저장된 전략 설정 그대로.
        "configs": [
            {
                "strategy_id": sid,
                "pool": entry["pool"],
                "ticker": entry["ticker"],
                "name": entry["name"],
                "trigger_pct": entry["trigger_pct"],
                "round_quantity": entry["round_quantity"],
                "rounds": entry["rounds"],
            }
            for sid, entry in mine.items()
        ],
        "strategies": _build_views(mine, fresh=False, include_trade_history=True),
    }


def _cash_summary(account_id: str) -> dict[str, Any]:
    """화면 상단의 현금 표시 — 원화 예수금과 계좌 평가액 대비 비율(%)."""
    state = _load_account_state(account_id)
    total = state["total_krw"]
    return {
        "cash_krw": state["cash_krw"],
        "pct": round(state["cash_krw"] / total * 100.0, 2) if total > 0 else None,
    }
