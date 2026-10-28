"""전략 사고팔기 운용 현황 조회 서비스 (UI/API 용).

전략 여러 개를 독립 운용한다. 각 전략은 **자기 종목 하나만** 설정된 회차 수까지 분할 매수한다.
전략은 종목풀(``pool``) 단위로 묶이고, 사고팔기를 켠 풀(종목풀 설정의 토글)에서만 만든다.
파라미터(매매 간격 %·회차당 수량·회차 수)는
DB(``system_config.strategy_trade_settings.strategies``)가 단일 소스이며 화면에서
편집한다. 슬랙 알람은 그 종목풀 슬리브가 붙은 계좌의 합성 슬랙 스위치를 따른다.

보유 현황은 백테스트가 아니라 그 종목풀의 사고팔기 **합성 슬리브가 붙은 계좌**의 **실제 보유 수량·평균단가**를
읽어 계산한다. 배치·원장은 없다 — 화면을 열 때마다 즉시 역산한다.

회차 역산 규칙 (자세한 근거는 strategy_trade_config 모듈 설명 참고)
---------------------------------------------------------------
- 진입 회차 수 m = 보유 수량 ÷ 회차당 수량 (반올림, 1~회차 수로 제한).
- 1호가 P1 = 평균단가 × m ÷ (1 + r + … + r^(m-1)),  r = 1 − 간격%.
- k호 진입가 = P1 × r^(k-1) · 매도 지정가 = 진입가 × (1 + 간격%).
- 다음 매수 = 마지막 매수가 × r. 보유가 없으면 1호 진입 = 전일 종가 × r,
  장중에 못 닿았으면 15:00(``FIRST_ENTRY_CLOSE_TIME``)부터 마감 전까지 현재가로 1호 매수.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from core.strategy.trade import rules as ladder
from utils.strategy_trade_config import (
    EDITABLE_CONFIG_KEYS,
    FIRST_ENTRY_CLOSE_TIME,
    validate_strategy_trade_config,
)

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def load_settings() -> dict[str, Any]:
    """저장된 슬랙 스위치 + 전략별 파라미터를 반환한다.

    전략 항목이 없거나 깨졌으면 시드값으로 슬쩍 넘어가지 않고 에러를 낸다 —
    그럴듯한 값이 화면에 떴다가 그대로 저장돼 실제 설정이 덮어써지는 것을 막는다.

    반환: ``{"strategies": {sid: {"pool", "ticker", "name", "trigger_pct",
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
        if not pool or not ticker or not name:
            raise RuntimeError(f"'{sid}' 전략의 종목풀·종목이 없습니다 — 화면에서 골라 저장하세요.")
        try:
            strategies[sid] = {
                "pool": pool,
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


def _validated_target(pool: str, config: dict[str, Any]) -> dict[str, Any]:
    """편집 입력에서 종목·파라미터를 검증해 저장 값으로 돌려준다. 실패 시 ValueError."""
    if not isinstance(config, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")
    etf = resolve_etf(pool, str(config.get("ticker") or ""))
    return {
        "pool": pool,
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
    # 종목이 바뀌면 그 전략의 알림 중복 방지 기록을 비운다 — 이전 보유 상태로 새 신호를
    # 판정하면 엉뚱한 확인·재알림이 나간다.
    if previous is not None and previous["ticker"] != target["ticker"]:
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
    # 다른 풀의 전략이 쓰는 종목 — 같은 보유로 회차를 두 번 세지 않게 풀 밖과도 겹치면 안 된다.
    taken = {entry["ticker"] for sid, entry in stored.items() if sid not in current}

    planned: list[tuple[str | None, dict[str, Any]]] = []
    for row in rows:
        sid = str(row.get("strategy_id") or "").strip() or None
        if sid is not None and sid not in current:
            raise ValueError(f"이 종목풀의 전략이 아닙니다: {sid}")
        target = _validated_target(pool, row)
        if target["ticker"] in taken:
            raise ValueError(f"{target['name']}({target['ticker']}) 전략이 이미 있습니다.")
        taken.add(target["ticker"])
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


def sleeve_for_pool(pool: str) -> dict[str, Any] | None:
    """그 종목풀의 사고팔기 합성 슬리브가 붙은 계좌와 한도 — 없으면 None.

    사고팔기는 계좌를 따로 고르지 않는다. 합성에서 `사고팔기 + 종목풀` 슬리브를 붙인 계좌가
    운용 계좌이고, 슬리브 비중 × 합성 기준금액이 그 종목풀 전략 전체의 매수 한도다.
    """
    from utils.account_settings_store import load_account_docs, mix_sleeves_of
    from utils.pool_strategy_use import STRATEGY_TRADE

    found = [
        (doc, sleeve)
        for doc in load_account_docs()
        for sleeve in mix_sleeves_of(doc)
        if sleeve["strategy"] == STRATEGY_TRADE and sleeve["pool"] == pool
    ]
    if not found:
        return None
    if len(found) > 1:
        raise RuntimeError(f"{pool} 종목풀의 사고팔기 슬리브가 여러 계좌에 붙어 있습니다.")
    doc, sleeve = found[0]
    capital = float(doc.get("mix_capital_krw") or 0)
    if capital <= 0:
        raise RuntimeError(f"'{doc['account_id']}' 계좌의 합성 운용 기준금액(KRW)이 설정돼 있지 않습니다.")
    return {
        "account_id": doc["account_id"],
        "account_name": str(doc.get("name") or doc["account_id"]),
        "weight_pct": float(sleeve["weight_pct"]),
        "limit_krw": capital * float(sleeve["weight_pct"]) / 100.0,
        # 슬랙 알람은 합성의 계좌 단위 스위치를 따른다 — 사고팔기에 따로 두지 않는다.
        "slack_enabled": bool(doc.get("mix_slack_enabled")),
    }


def strategy_budget(sleeve: dict[str, Any], strategy_count: int) -> float:
    """전략 하나의 매수 한도(원) — 슬리브 한도를 전략 수로 균등 분할한다."""
    if strategy_count <= 0:
        raise ValueError("전략이 없어 한도를 나눌 수 없습니다.")
    return float(sleeve["limit_krw"]) / strategy_count


def _load_account_rows(account_id: str) -> dict[str, dict[str, float]]:
    """계좌에서 보유 종목의 수량·평균단가·현재가를 읽는다 (전 전략 공용, 1회 조회).

    ``load_real_holdings_table`` 은 실시간 시세를 반영한 현재가까지 채워준다. 보유가 하나도 없는
    계좌는 빈 결과다 — 모든 전략이 1호 진입 대기로 나온다.
    """
    from utils.portfolio_io import load_real_holdings_table

    frame = load_real_holdings_table(account_id)
    if frame is None or frame.empty:
        return {}

    ticker_column = "ticker" if "ticker" in frame.columns else "티커"
    required = {ticker_column, "수량", "평균 매입가", "현재가"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"보유 테이블에 필요한 컬럼이 없습니다: {', '.join(sorted(missing))}")

    rows: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        ticker = str(row[ticker_column]).strip()
        if not ticker:
            continue
        rows[ticker] = {
            "quantity": float(row["수량"] or 0),
            "avg_price": float(row["평균 매입가"] or 0),
            "close": float(row["현재가"] or 0),
        }
    return rows


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


def _build_strategy_view(
    strategy_id: str,
    strategy_settings: dict[str, Any],
    account_rows: dict[str, dict[str, float]],
    budget: float,
) -> dict[str, Any]:
    """전략 하나의 운용 현황(회차 표·상태)을 계좌 보유에서 역산한다. ``budget`` 은 이 전략의 매수 한도(원)."""
    pool = str(strategy_settings["pool"])
    ticker = str(strategy_settings["ticker"])
    name = str(strategy_settings["name"])
    trigger_pct = float(strategy_settings["trigger_pct"])
    round_quantity = int(strategy_settings["round_quantity"])
    max_rounds = int(strategy_settings["rounds"])

    account = account_rows.get(ticker) or {}
    quantity = int(account.get("quantity") or 0)
    avg_price = float(account.get("avg_price") or 0)
    live_close = float(account.get("close") or 0) or _load_unheld_closes([ticker], pool).get(ticker)
    index_status = _load_ticker_status(pool, ticker, name, trigger_pct, live_close)
    close = float(index_status["close"])
    # 보유 0 인데 장중 1호 진입선에 못 닿은 채 장 마감이 다가오면 현재가로 1호를 산다.
    close_entry = False

    # ── 회차 역산 — 수량으로 진입 회차 수, 평균단가로 1호가 ──
    held_rounds = 0
    first_entry: float | None = None
    quantity_mismatch = False
    if quantity > 0 and avg_price > 0:
        held_rounds = max(1, min(max_rounds, round(quantity / round_quantity)))
        quantity_mismatch = quantity != held_rounds * round_quantity
        first_entry = ladder.first_entry_from_average(avg_price, held_rounds, trigger_pct)
    else:
        close_entry = _first_entry_close_window()

    def entry_price(round_no: int) -> float | None:
        if first_entry is None:
            return None
        return ladder.entry_price(first_entry, round_no, trigger_pct)

    # 이미 산 금액 — 계좌의 실제 매입 원가(수량 × 평균단가). 다음 회차가 한도를 넘는지 이걸로 본다.
    invested = quantity * avg_price if held_rounds else 0.0

    rows: list[dict[str, Any]] = []
    for round_no in range(1, max_rounds + 1):
        held = round_no <= held_rounds
        is_next = round_no == held_rounds + 1
        entry = entry_price(round_no)
        if held and entry is not None:
            sell_limit = ladder.sell_limit(entry, trigger_pct)
            rows.append(
                {
                    "round": round_no,
                    "fixed_round": round_no,
                    "ticker": ticker,
                    "name": name,
                    "held": True,
                    "avg_price": round(entry),
                    "close": round(close) if close > 0 else None,
                    "profit_pct": round((close / entry - 1.0) * 100.0, 2) if close > 0 else None,
                    "buy_limit": None,
                    "buy_index": round(entry),  # 이미 산 회차 — 진입가 자리를 참고로 남긴다
                    "sell_limit": sell_limit,
                    "sell_index": sell_limit,
                    "sell_reached": bool(close > 0 and close >= sell_limit),
                    "buy_reached": False,
                    "close_entry": False,
                    "over_limit": False,
                    "is_next": False,
                }
            )
            continue
        # 미보유 회차 — 사다리 자리(보유가 있으면 1호가 기준, 없으면 전일 종가 기준).
        # 지정가로 바로 쓸 수 있게 호가 단위로 반올림한다.
        if first_entry is not None:
            buy_limit = ladder.buy_limit(first_entry, round_no, trigger_pct)
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
                "avg_price": None,
                "close": round(close) if close > 0 else None,
                "profit_pct": None,
                "buy_limit": buy_limit,
                "buy_index": buy_limit,
                "sell_limit": None,
                "sell_index": None,
                "sell_reached": False,
                "buy_reached": bool(is_next and within_budget and close > 0 and (close <= buy_limit or close_entry)),
                # 진입선이 아니라 장 마감 매수로 닿은 1호 — 지정가 대신 현재가로 산다.
                "close_entry": bool(market_entry and within_budget),
                # 다음 회차가 매수 한도를 넘는다 — 이 회차부터는 사지 않는다.
                "over_limit": bool(is_next and not within_budget),
                "is_next": is_next,
            }
        )

    # 다음 회차 매수까지 남은 하락률(%) — 도달했으면 0 이상(+). 회차 소진이면 None.
    next_buy_drop_pct = None
    next_row = next((r for r in rows if r["is_next"]), None)
    if next_row is not None and next_row["buy_limit"] is not None and close > 0:
        next_buy_drop_pct = round((next_row["buy_limit"] / close - 1.0) * 100.0, 2)

    # 가장 가까운 매도 = 마지막(가장 싼) 회차 — 사다리 구조상 항상 먼저 도달한다.
    next_sell_round = None
    next_sell_rise_pct = None
    held_rows = [r for r in rows if r["held"] and r["sell_limit"] is not None and close > 0]
    if held_rows:
        deepest = held_rows[-1]
        next_sell_round = deepest["round"]
        next_sell_rise_pct = round((deepest["sell_limit"] / close - 1.0) * 100.0, 2)

    last_entry = entry_price(held_rounds) if held_rounds else None
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
            "last_buy_price": round(last_entry) if last_entry is not None else None,
            "next_buy_drop_pct": next_buy_drop_pct,
            "next_sell_round": next_sell_round,
            "next_sell_rise_pct": next_sell_rise_pct,
            # 보유 수량이 회차×수량과 다르면 역산이 근사가 된다 — 화면이 경고를 띄운다.
            "quantity_mismatch": quantity_mismatch,
            "held_quantity": quantity,
            # 매수 한도(원) · 이미 산 금액 · 다음 회차가 한도에 막혔는지.
            "budget": round(budget),
            "invested": round(invested),
            "limit_blocked": bool(next_row is not None and next_row["over_limit"]),
        },
        "rounds": rows,
    }


def _build_views(strategies: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """전략들의 운용 현황 — 풀마다 슬리브가 붙은 계좌의 보유를 읽고, 계좌 조회는 계좌당 1회다.

    슬리브가 없는 풀의 전략은 계좌를 알 수 없어 만들 수 없다 — 명시적 오류로 알린다.
    """
    count_by_pool: dict[str, int] = {}
    for entry in strategies.values():
        count_by_pool[entry["pool"]] = count_by_pool.get(entry["pool"], 0) + 1
    sleeves: dict[str, dict[str, Any]] = {}
    for pool in count_by_pool:
        sleeve = sleeve_for_pool(pool)
        if sleeve is None:
            raise ValueError(f"{pool} 종목풀의 사고팔기 슬리브가 어느 계좌에도 붙어 있지 않습니다 — 합성에서 붙이세요.")
        sleeves[pool] = sleeve
    rows_by_account = {
        account_id: _load_account_rows(account_id) for account_id in {s["account_id"] for s in sleeves.values()}
    }
    return [
        _build_strategy_view(
            sid,
            entry,
            rows_by_account[sleeves[entry["pool"]]["account_id"]],
            strategy_budget(sleeves[entry["pool"]], count_by_pool[entry["pool"]]),
        )
        for sid, entry in strategies.items()
    ]


def load_pool_views(pool: str) -> list[dict[str, Any]]:
    """한 종목풀의 전략 운용 현황 — 합성 슬리브가 보유 표시에 쓴다."""
    in_pool = {sid: entry for sid, entry in load_settings()["strategies"].items() if entry["pool"] == pool}
    if not in_pool:
        raise ValueError(f"{pool} 종목풀에 사고팔기 전략이 없습니다.")
    return _build_views(in_pool)


def load_views_for_pools(pools: list[str]) -> dict[str, Any]:
    """주어진 종목풀들의 전략 운용 현황 — 알림용(화면은 풀 하나씩 본다)."""
    wanted = set(pools)
    return {
        "strategies": _build_views(
            {sid: entry for sid, entry in load_settings()["strategies"].items() if entry["pool"] in wanted}
        )
    }


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
    sleeve = sleeve_for_pool(selected) if selected is not None else None
    return {
        "pool_options": pool_options(enabled),
        "pool": selected,
        "month_options": get_month_options(),
        # 합성 슬리브 — 운용 계좌와 매수 한도의 출처. 없으면 None(화면이 붙이라고 안내한다).
        "sleeve": (
            {**sleeve, "per_strategy_limit_krw": strategy_budget(sleeve, len(in_pool))}
            if sleeve is not None and in_pool
            else sleeve
        ),
        # 편집 표의 행 — 슬리브 유무와 무관하게 저장된 전략 설정 그대로.
        "configs": [
            {
                "strategy_id": sid,
                "ticker": entry["ticker"],
                "name": entry["name"],
                "trigger_pct": entry["trigger_pct"],
                "round_quantity": entry["round_quantity"],
                "rounds": entry["rounds"],
            }
            for sid, entry in in_pool.items()
        ],
        "strategies": _build_views(in_pool) if sleeve is not None else [],
    }
