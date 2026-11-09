"""전략 사고팔기 운용 현황 조회 서비스 (UI/API 용).

전략 여러 개를 독립 운용한다. 각 전략은 **자기 종목 하나만** 설정된 회차 수까지 분할 매수한다.
전략은 종목풀(``pool``) 단위로 묶이고, 사고팔기를 켠 풀(종목풀 설정의 토글)에서만 만든다.
파라미터(매매 간격 %·회차당 수량·회차 수)는
DB(``system_config.strategy_trade_settings.strategies``)가 단일 소스이며 화면에서
편집한다. 슬랙 알람은 그 종목풀 슬리브가 붙은 계좌의 합성 슬랙 스위치를 따른다.

운용 계좌는 전략마다 고른다 — 그 풀의 사고팔기 **합성 슬리브가 붙은 계좌** 중 하나다. 같은 풀을 여러 계좌가
쓸 수 있고, 매수 한도는 (풀, 계좌) 슬리브 한도를 그 계좌의 전략 수로 나눈다.
보유 현황은 백테스트가 아니라 전략의 운용 계좌의 **증권사 체결 이력**에서 만든다.
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
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from core.strategy.trade import rules as ladder
from utils.strategy_trade_config import (
    EDITABLE_CONFIG_KEYS,
    FIRST_ENTRY_CLOSE_TIME,
    validate_strategy_trade_config,
)
from utils.strategy_trade_ledger import Ledger, LedgerError, forget_ledgers, read_ledgers, refresh_ledgers

KST = ZoneInfo("Asia/Seoul")
_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def load_settings() -> dict[str, Any]:
    """저장된 슬랙 스위치 + 전략별 파라미터를 반환한다.

    전략 항목이 없거나 깨졌으면 시드값으로 슬쩍 넘어가지 않고 에러를 낸다 —
    그럴듯한 값이 화면에 떴다가 그대로 저장돼 실제 설정이 덮어써지는 것을 막는다.

    반환: ``{"strategies": {sid: {"pool", "account_id", "ticker", "name", "trigger_pct",
    "round_quantity", "rounds"}}}`` — 풀·종목·운용 계좌는 전략마다 따로이고, 전략 목록은 DB 가 정한다.
    """
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패해 전략 사고팔기 설정을 읽을 수 없습니다.")
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    stored = doc.get("strategies")
    if not isinstance(stored, dict):
        raise RuntimeError(
            f"저장된 전략 사고팔기 설정이 없습니다 "
            f"({_CONFIG_COLLECTION}.{_SETTINGS_KEY}.strategies 를 먼저 저장하세요)."
        )

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
                # 운용 계좌 — 전략마다 고른다. 그 계좌의 체결로 회차를 세고 그 계좌로 주문한다.
                "account_id": account_id,
                "ticker": ticker,
                "name": name,
                **validate_strategy_trade_config({key: entry.get(key) for key in EDITABLE_CONFIG_KEYS}),
            }
        except ValueError as error:
            raise ValueError(f"'{sid}' 전략 설정이 올바르지 않습니다: {error}") from error

    return {"strategies": strategies}


def _require_enabled_pool(pool: str) -> None:
    """종목풀 설정에서 이 풀의 사고팔기가 켜져 있어야 한다 — 꺼져 있으면 종목을 고를 수 없다."""
    from utils.pool_strategy_use import STRATEGY_TRADE, pools_using

    if pool not in pools_using(STRATEGY_TRADE):
        raise ValueError(f"{pool} 종목풀에서 사고팔기가 꺼져 있습니다 — 종목풀 설정에서 먼저 켜세요.")


def etf_options(pool: str) -> list[dict[str, str]]:
    """전략에 쓸 수 있는 종목 — 그 종목풀의 ETF(티커·이름)."""
    from utils.stock_list_io import get_etfs

    _require_enabled_pool(pool)
    options = [
        {"ticker": str(item.get("ticker") or "").strip(), "name": str(item.get("name") or "").strip()}
        for item in get_etfs(pool) or []
    ]
    return [option for option in options if option["ticker"] and option["name"]]


def resolve_etf(pool: str, ticker: str) -> dict[str, str]:
    """티커로 전략에 쓸 ETF 를 찾는다 — 그 종목풀에 있는 것만. 없으면 ValueError."""
    wanted = str(ticker or "").strip().upper()
    if not wanted:
        raise ValueError("티커를 입력하세요.")
    for option in etf_options(pool):
        if option["ticker"].upper() == wanted:
            return option
    raise ValueError(f"{pool} 종목풀에 없는 종목입니다: {ticker} — 종목풀에 먼저 추가하세요.")


def _validated_target(pool: str, config: dict[str, Any], account_ids: set[str]) -> dict[str, Any]:
    """편집 입력에서 운용 계좌·종목·파라미터를 검증해 저장 값으로 돌려준다. 실패 시 ValueError.

    ``account_ids`` 는 이 풀에 사고팔기 슬리브가 붙은 계좌 — 운용 계좌는 그중에서만 고른다.
    """
    if not isinstance(config, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")
    etf = resolve_etf(pool, str(config.get("ticker") or ""))
    account_id = str(config.get("account_id") or "").strip()
    if account_id not in account_ids:
        raise ValueError(
            f"{etf['name']}({etf['ticker']}): 운용 계좌는 이 종목풀의 사고팔기 슬리브가 붙은 계좌에서 고르세요 "
            f"({account_id or '미선택'})."
        )
    return {
        "pool": pool,
        "account_id": account_id,
        "ticker": etf["ticker"],
        "name": etf["name"],
        **validate_strategy_trade_config(config),
    }


def _save_strategy(strategy_id: str, target: dict[str, Any], *, previous: dict[str, Any] | None) -> None:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    update: dict[str, Any] = {
        "$set": {
            "updated_at": datetime.now().isoformat(),
            **{f"strategies.{strategy_id}.{k}": v for k, v in target.items()},
        }
    }
    # 종목이나 운용 계좌가 바뀌면 그 전략의 알림 중복 방지 기록을 비운다 — 이전 보유 상태로 새 신호를
    # 판정하면 엉뚱한 확인·재알림이 나간다.
    if previous is not None and (
        previous["ticker"] != target["ticker"] or previous["account_id"] != target["account_id"]
    ):
        update["$unset"] = _sent_log_unset(db, strategy_id)
        if not update["$unset"]:
            del update["$unset"]
    db[_CONFIG_COLLECTION].update_one({"_id": _SETTINGS_KEY}, update, upsert=True)


def _sent_log_unset(db: Any, strategy_id: str) -> dict[str, str]:
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}, {"sent_log": 1}) or {}
    return {f"sent_log.{key}": "" for key in (doc.get("sent_log") or {}) if str(key).startswith(f"{strategy_id}:")}


def _delete_strategy(strategy_id: str) -> None:
    """전략 하나와 그 알림 중복 방지 기록을 지운다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    unset = {f"strategies.{strategy_id}": "", **_sent_log_unset(db, strategy_id)}
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY}, {"$set": {"updated_at": datetime.now().isoformat()}, "$unset": unset}
    )


def save_pool_strategies(pool: str, rows: list[dict[str, Any]]) -> None:
    """한 종목풀의 전략 목록을 통째로 저장한다 — 전략 id 가 있는 행은 갱신, 없는 행은 추가, 빠진 전략은 삭제.

    검증은 저장 전에 전부 끝낸다 — 한 행이라도 틀리면 아무것도 바꾸지 않는다.
    """
    from uuid import uuid4

    _require_enabled_pool(pool)
    if not isinstance(rows, list):
        raise ValueError("전략 목록 형식이 올바르지 않습니다.")
    stored = load_settings()["strategies"]
    current = {sid: entry for sid, entry in stored.items() if entry["pool"] == pool}
    account_ids = {sleeve["account_id"] for sleeve in sleeves_for_pool(pool)}
    # 같은 계좌의 (계좌, 종목)은 하나뿐이다 — 회차를 그 계좌의 종목 체결로 세므로 겹치면 두 번 센다.
    # 다른 풀의 전략과도 겹치면 안 된다. 계좌가 다르면 같은 종목도 된다.
    taken = {(entry["account_id"], entry["ticker"]) for sid, entry in stored.items() if sid not in current}

    planned: list[tuple[str | None, dict[str, Any]]] = []
    for row in rows:
        sid = str(row.get("strategy_id") or "").strip() or None
        if sid is not None and sid not in current:
            raise ValueError(f"이 종목풀의 전략이 아닙니다: {sid}")
        target = _validated_target(pool, row, account_ids)
        key = (target["account_id"], target["ticker"])
        if key in taken:
            raise ValueError(
                f"{target['account_id']} 계좌에 {target['name']}({target['ticker']}) 전략이 이미 있습니다."
            )
        taken.add(key)
        planned.append((sid, target))

    kept = {sid for sid, _ in planned if sid is not None}
    for sid, target in planned:
        if sid is None:
            _save_strategy(f"s{uuid4().hex[:8]}", target, previous=None)
        else:
            _save_strategy(sid, target, previous=current[sid])
    for sid in current:
        if sid not in kept:
            _delete_strategy(sid)


def delete_pool_strategies(pool: str) -> int:
    """그 종목풀의 사고팔기 전략을 모두 지운다(알림 기록 포함). 지운 개수를 반환한다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    stored = (db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}).get("strategies") or {}
    doomed = [sid for sid, entry in stored.items() if isinstance(entry, dict) and entry.get("pool") == pool]
    for sid in doomed:
        _delete_strategy(sid)
    return len(doomed)


def sleeves_for_pool(pool: str) -> list[dict[str, Any]]:
    """그 종목풀의 사고팔기 합성 슬리브가 붙은 계좌들과 한도 — 전략이 고를 수 있는 운용 계좌 목록(계좌 순서).

    슬리브 비중 × 합성 운용 기준금액(계좌 평가액)이 그 계좌에서 이 풀 전략들의 매수 한도다.
    """
    from utils.account_settings_store import load_live_account_docs, mix_sleeves_of
    from utils.cash_model import currency_for_country
    from utils.pool_strategy_use import STRATEGY_TRADE
    from utils.strategy_mix_service import operating_capital_krw

    found: list[dict[str, Any]] = []
    for doc in load_live_account_docs():
        for sleeve in mix_sleeves_of(doc):
            if sleeve["strategy"] != STRATEGY_TRADE or sleeve["pool"] != pool:
                continue
            capital = operating_capital_krw(doc["account_id"], currency_for_country(doc["country_code"]))
            found.append(
                {
                    "account_id": doc["account_id"],
                    "account_name": str(doc.get("name") or doc["account_id"]),
                    "weight_pct": float(sleeve["weight_pct"]),
                    "limit_krw": capital * float(sleeve["weight_pct"]) / 100.0,
                    # 슬랙 알람은 합성의 계좌 단위 스위치를 따른다 — 사고팔기에 따로 두지 않는다.
                    "slack_enabled": bool(doc.get("mix_slack_enabled")),
                }
            )
    return found


def sleeve_of(pool: str, account_id: str) -> dict[str, Any]:
    """그 계좌에 붙은 그 풀의 사고팔기 슬리브 — 없으면 에러(합성에서 슬리브를 뗐는데 전략이 남은 경우)."""
    found = next((sleeve for sleeve in sleeves_for_pool(pool) if sleeve["account_id"] == account_id), None)
    if found is None:
        raise ValueError(
            f"'{account_id}' 계좌에 {pool} 종목풀의 사고팔기 슬리브가 없습니다 — 합성에서 붙이거나 전략의 계좌를 바꾸세요."
        )
    return found


def broker_account_no_for_account(account_id: str) -> str:
    """사고팔기 운용 계좌에 연동된 증권사 계좌번호 — 연동이 없으면 에러."""
    from utils.account_settings_store import load_live_account_docs

    doc = next((doc for doc in load_live_account_docs() if doc["account_id"] == account_id), None)
    if doc is None:
        raise ValueError(f"'{account_id}' 계좌가 없거나 운영 계좌가 아닙니다.")
    account_no = str((doc.get("broker_api") or {}).get("account_no") or "").strip()
    if not account_no:
        raise ValueError(f"'{account_id}' 계좌에 사고팔기 주문용 증권사 API(나무증권) 연동이 저장돼 있지 않습니다.")
    return account_no


def strategy_budget(sleeve: dict[str, Any], strategy_count: int) -> float:
    """전략 하나의 매수 한도(원) — 슬리브 한도를 전략 수로 균등 분할한다."""
    if strategy_count <= 0:
        raise ValueError("전략이 없어 한도를 나눌 수 없습니다.")
    return float(sleeve["limit_krw"]) / strategy_count


def _load_account_closes(account_id: str) -> dict[str, float]:
    """계좌 보유 종목의 현재가 (전 전략 공용, 1회 조회).

    ``load_real_holdings_table`` 은 실시간 시세를 반영한 현재가까지 채워준다. 수량·평균단가는 쓰지 않는다 —
    보유는 증권사 체결 이력(회차 원장)이 정한다.
    """
    from utils.portfolio_io import load_real_holdings_table

    frame = load_real_holdings_table(account_id)
    if frame is None or frame.empty:
        return {}

    ticker_column = "ticker" if "ticker" in frame.columns else "티커"
    required = {ticker_column, "현재가"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"보유 테이블에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")

    return {
        str(row[ticker_column]).strip(): float(row["현재가"] or 0)
        for _, row in frame.iterrows()
        if str(row[ticker_column]).strip()
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
    pool: str, ticker: str, name: str, trigger_pct: float, live_close: float | None
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
        "buy_trigger": ladder.first_buy_limit(prev_close, float(trigger_pct)),
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


def _attach_today(rows: list[dict[str, Any]], ledger: Ledger, round_quantity: int, held_rounds: int) -> None:
    """회차 행에 오늘 체결(`fills`)과 걸려 있는 수동 주문(`manual_orders`)을 붙인다 — 표시용 기록이다.

    수동 매수는 다음 진입 회차에, 수동 매도는 가장 먼저 팔릴 맨 위 보유 회차에 붙는다.
    """
    today = datetime.now(KST).strftime("%Y-%m-%d")
    events = [event for event in ledger.round_events(round_quantity) if event["day"] == today]
    for row in rows:
        row["fills"] = [
            {field: event[field] for field in ("source", "side", "price", "quantity")}
            for event in events
            if event["round"] == row["round"]
        ]
        row["manual_orders"] = [
            order
            for order in ledger.manual_orders
            if (order["side"] == "buy" and row["is_next"]) or (order["side"] == "sell" and row["round"] == held_rounds)
        ]


def _build_strategy_view(
    strategy_id: str,
    strategy_settings: dict[str, Any],
    account_closes: dict[str, float],
    ledger: Ledger,
    budget: float,
) -> dict[str, Any]:
    """전략 하나의 운용 현황(회차 표·상태)을 회차 원장에서 만든다. ``budget`` 은 이 전략의 매수 한도(원)."""
    pool = str(strategy_settings["pool"])
    ticker = str(strategy_settings["ticker"])
    name = str(strategy_settings["name"])
    trigger_pct = float(strategy_settings["trigger_pct"])
    round_quantity = int(strategy_settings["round_quantity"])
    max_rounds = int(strategy_settings["rounds"])

    quantity = ledger.quantity
    live_close = account_closes.get(ticker) or _load_unheld_closes([ticker], pool).get(ticker)
    index_status = _load_ticker_status(pool, ticker, name, trigger_pct, live_close)
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
        held_rounds and close > 0 and close >= ladder.sell_limit(round_fills[held_rounds - 1]["price"], trigger_pct)
    )

    # 이미 산 금액 — 보유 로트의 실제 매입 원가. 다음 회차가 한도를 넘는지 이걸로 본다.
    invested = ledger.cost if held_rounds else 0.0

    rows: list[dict[str, Any]] = []
    for round_no in range(1, max_rounds + 1):
        held = round_no <= held_rounds
        is_next = round_no == held_rounds + 1
        if held:
            fill = round_fills[round_no - 1]
            sell_limit = ladder.sell_limit(fill["price"], trigger_pct)
            rows.append(
                {
                    "round": round_no,
                    "fixed_round": round_no,
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
                    "over_limit": False,
                    "is_next": False,
                }
            )
            continue
        # 미보유 회차 — 보유가 있으면 마지막 보유 체결가에서 내려간 자리(다음 회차 이후는 예상값),
        # 없으면 전일 종가 기준. 지정가로 바로 쓸 수 있게 호가 단위로 반올림한다.
        if last_fill is not None:
            buy_limit = ladder.projected_buy_limit(last_fill, round_no - held_rounds, trigger_pct)
        else:
            buy_limit = ladder.preview_buy_limit(float(index_status["prev_close"]), round_no, trigger_pct)
        # 15:00 시장가 매수는 현재가로 사므로 한도 검사도 그 가격으로 한다.
        market_entry = is_next and close_entry and close > buy_limit
        within_budget = not is_next or ladder.within_budget(
            invested, close if market_entry else buy_limit, round_quantity, budget
        )
        rows.append(
            {
                "round": round_no,
                "fixed_round": round_no,
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
                    is_next and not sell_pending and within_budget and close > 0 and (close <= buy_limit or close_entry)
                ),
                # 진입선이 아니라 장 마감 매수로 닿은 1호 — 지정가 대신 현재가로 산다.
                "close_entry": bool(market_entry and within_budget),
                # 다음 회차가 매수 한도를 넘는다 — 이 회차부터는 사지 않는다.
                "over_limit": bool(is_next and not within_budget),
                "is_next": is_next,
            }
        )

    _attach_today(rows, ledger, round_quantity, held_rounds)

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
            # 매수 한도(원) · 이미 산 금액 · 다음 회차가 한도에 막혔는지.
            "budget": round(budget),
            "invested": round(invested),
            "limit_blocked": bool(next_row is not None and next_row["over_limit"]),
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
    strategies = load_settings()["strategies"]
    failed: dict[str, str] = {}
    for account_id, tickers in _tickers_by_account(strategies).items():
        try:
            account_no = broker_account_no_for_account(account_id)
        except ValueError as error:
            failed.update({f"{account_id}:{ticker}": str(error) for ticker in tickers})
            continue
        for ticker, ledger in refresh_ledgers(account_no, tickers).items():
            if isinstance(ledger, LedgerError):
                failed[f"{account_id}:{ticker}"] = str(ledger)
        forget_ledgers(account_no, tickers)
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
    """계좌별 회차 원장 — 증권사 연동이 없는 계좌는 원장 대신 이유(문자열)다(그 계좌 전략만 실패로 본다)."""
    load = refresh_ledgers if fresh else read_ledgers
    ledgers: dict[str, dict[str, Ledger | LedgerError] | str] = {}
    for account_id, tickers in tickers_by_account.items():
        try:
            account_no = broker_account_no_for_account(account_id)
        except ValueError as error:
            ledgers[account_id] = str(error)
            continue
        ledgers[account_id] = load(account_no, tickers)
    return ledgers


def _build_views(strategies: dict[str, dict[str, Any]], *, fresh: bool) -> list[dict[str, Any]]:
    """전략들의 운용 현황 — 전략마다 고른 운용 계좌의 회차 원장으로 만든다.

    ``fresh`` 가 참이면 원장을 증권사에서 새로 계산해 저장하고(느리다), 아니면 배치가 저장해 둔 것을 읽는다.

    매수 한도는 (풀, 계좌) 슬리브 한도를 그 계좌에서 그 풀을 쓰는 전략 수로 균등 분할한다.
    계좌에 그 풀 슬리브가 없거나(합성에서 뗐는데 전략이 남음) 증권사 연동이 없거나 회차 원장을 못 만든 전략은
    `error` 만 담은 항목으로 돌려준다(`_failed_view`) — 한 전략의 문제가 다른 전략을 막지 않는다.
    """
    by_pool = {
        pool: {sleeve["account_id"]: sleeve for sleeve in sleeves_for_pool(pool)}
        for pool in {entry["pool"] for entry in strategies.values()}
    }
    count_by_key: dict[tuple[str, str], int] = {}
    for entry in strategies.values():
        key = (entry["pool"], entry["account_id"])
        count_by_key[key] = count_by_key.get(key, 0) + 1
    tickers_by_account = _tickers_by_account(strategies)
    ledgers = _account_ledgers(tickers_by_account, fresh=fresh)
    closes_by_account = {
        account_id: _load_account_closes(account_id)
        for account_id, account_ledgers in ledgers.items()
        if not isinstance(account_ledgers, str)
    }
    from utils.account_settings_store import load_account_docs

    names = {doc["account_id"]: str(doc.get("name") or doc["account_id"]) for doc in load_account_docs()}
    views: list[dict[str, Any]] = []
    for sid, entry in strategies.items():
        account_id = entry["account_id"]
        sleeve = by_pool[entry["pool"]].get(account_id)
        account_ledgers = ledgers[account_id]
        if sleeve is None:
            view = _failed_view(
                sid,
                entry,
                f"'{account_id}' 계좌에 이 종목풀의 사고팔기 슬리브가 없습니다 — 합성에서 붙이거나 전략의 계좌를 바꾸세요.",
            )
        elif isinstance(account_ledgers, str):
            view = _failed_view(sid, entry, account_ledgers)
        elif isinstance(ledger := account_ledgers[entry["ticker"]], LedgerError):
            view = _failed_view(sid, entry, ledger)
        else:
            view = _build_strategy_view(
                sid,
                entry,
                closes_by_account[account_id],
                ledger,
                strategy_budget(sleeve, count_by_key[(entry["pool"], account_id)]),
            )
        # 운용 계좌 — 화면 카드 머리·슬랙 문구·주문 계좌가 이것을 쓴다.
        views.append({**view, "account_id": account_id, "account_name": names.get(account_id, account_id)})
    return views


def load_account_pool_views(pool: str, account_id: str) -> list[dict[str, Any]]:
    """한 계좌의 한 종목풀 전략 운용 현황 — 합성 슬리브가 보유 표시에 쓴다. 회차 원장을 못 만든 전략이 있으면 오류다."""
    selected = {
        sid: entry
        for sid, entry in load_settings()["strategies"].items()
        if entry["pool"] == pool and entry["account_id"] == account_id
    }
    if not selected:
        raise ValueError(
            f"'{account_id}' 계좌에 {pool} 종목풀의 사고팔기 전략이 없습니다 — 사고팔기 화면에서 계좌를 고르세요."
        )
    views = _build_views(selected, fresh=False)
    failed = [view for view in views if "error" in view]
    if failed:
        raise ValueError(f"{failed[0]['label']}: {failed[0]['error']}")
    return views


def load_views_where(keep: Callable[[dict[str, Any]], bool], *, fresh: bool) -> dict[str, Any]:
    """``keep(전략 설정)`` 이 참인 전략들의 운용 현황 — 알림·주문용. ``fresh`` 는 `_build_views` 와 같다.

    회차 원장을 못 만든 전략은 `strategies` 에서 빼고 `errors`(전략 id → 이유)에 담는다 — 그 전략만 건너뛴다.
    """
    views = _build_views(
        {sid: entry for sid, entry in load_settings()["strategies"].items() if keep(entry)},
        fresh=fresh,
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


def load_strategy_trade_view(pool: str | None) -> dict[str, Any]:
    """화면용 — 사고팔기를 켠 종목풀 목록과, 고른 풀의 전략 설정·운용 현황·합성 슬리브.

    ``pool`` 이 None 이면 켜진 풀 중 첫 번째를 보여준다(다른 전략 화면과 같다). 켜진 풀이 없으면
    ``pool`` 은 None 이다. 슬리브가 안 붙은 풀은 설정(``configs``)만 주고 운용 현황은 비운다.
    """
    from utils.pool_signal_backtest_service import get_month_options
    from utils.pool_strategy_use import STRATEGY_TRADE, pool_options, pools_using

    settings = load_settings()
    enabled = pools_using(STRATEGY_TRADE)
    if pool is not None and pool not in enabled:
        raise ValueError(f"{pool} 종목풀에서 사고팔기가 꺼져 있습니다 — 종목풀 설정에서 먼저 켜세요.")
    selected = pool if pool is not None else (enabled[0] if enabled else None)
    in_pool = {sid: entry for sid, entry in settings["strategies"].items() if entry["pool"] == selected}
    sleeves = sleeves_for_pool(selected) if selected is not None else []
    return {
        "pool_options": pool_options(enabled),
        "pool": selected,
        "month_options": get_month_options(),
        # 합성 슬리브 — 전략이 고를 수 있는 운용 계좌와 계좌별 매수 한도의 출처. 비면 화면이 붙이라고 안내한다.
        "sleeves": [_sleeve_with_split(sleeve, in_pool) for sleeve in sleeves],
        # 편집 표의 행 — 슬리브 유무와 무관하게 저장된 전략 설정 그대로.
        "configs": [
            {
                "strategy_id": sid,
                "account_id": entry["account_id"],
                "ticker": entry["ticker"],
                "name": entry["name"],
                "trigger_pct": entry["trigger_pct"],
                "round_quantity": entry["round_quantity"],
                "rounds": entry["rounds"],
            }
            for sid, entry in in_pool.items()
        ],
        "strategies": _build_views(in_pool, fresh=False),
    }


def _sleeve_with_split(sleeve: dict[str, Any], in_pool: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """계좌 슬리브에 전략당 한도를 붙인다 — 그 계좌를 고른 전략 수로 균등 분할. 고른 전략이 없으면 None."""
    count = sum(1 for entry in in_pool.values() if entry["account_id"] == sleeve["account_id"])
    return {
        **sleeve,
        "strategy_count": count,
        "per_strategy_limit_krw": strategy_budget(sleeve, count) if count else None,
    }
