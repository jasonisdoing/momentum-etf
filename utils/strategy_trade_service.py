"""전략 사고팔기 운용 현황 조회 서비스 (UI/API 용).

전략 3개(KODEX 200 / KODEX 코스닥150 / KIWOOM 삼성SK그룹TOP4+)를 같은 계좌에서
독립 운용한다. 각 전략은 **자기 종목 하나만** 설정된 회차 수까지 분할 매수한다.
파라미터(매매 간격 %·회차당 수량·회차 수)는
DB(``system_config.strategy_trade_settings.strategies``)가 단일 소스이며 화면에서
편집한다. 슬랙 스위치는 전역 1개다.

보유 현황은 백테스트가 아니라 설정에서 고른 계좌(``account_id``)의 **실제 보유 수량·평균단가**를
읽어 계산한다. 배치·원장은 없다 — 화면을 열 때마다 즉시 역산한다.

회차 역산 규칙 (자세한 근거는 strategy_trade_config 모듈 설명 참고)
---------------------------------------------------------------
- 진입 회차 수 m = 보유 수량 ÷ 회차당 수량 (반올림, 1~회차 수로 제한).
- 1호가 P1 = 평균단가 × m ÷ (1 + r + … + r^(m-1)),  r = 1 − 간격%.
- k호 진입가 = P1 × r^(k-1) · 매도 지정가 = 진입가 × (1 + 간격%).
- 다음 매수 = 마지막 매수가 × r. 보유가 없으면 1호 진입 = 전일 종가 × r.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.strategy_trade_config import (
    EDITABLE_CONFIG_KEYS,
    ETF_POOL,
    PRICE_CACHE_TICKER_TYPES,
    validate_strategy_trade_config,
)

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def load_settings() -> dict[str, Any]:
    """저장된 슬랙 스위치 + 전략별 파라미터를 반환한다.

    전략 항목이 없거나 깨졌으면 시드값으로 슬쩍 넘어가지 않고 에러를 낸다 —
    그럴듯한 값이 화면에 떴다가 그대로 저장돼 실제 설정이 덮어써지는 것을 막는다.

    반환: ``{"slack_enabled", "strategies": {sid: {"account_id", "ticker", "name", "trigger_pct",
    "round_quantity", "rounds"}}}`` — 종목·운용 계좌는 전략마다 따로이고, 전략 목록은 DB 가 정한다.
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
        account_id = str(entry.get("account_id") or "").strip()
        ticker = str(entry.get("ticker") or "").strip()
        name = str(entry.get("name") or "").strip()
        if not account_id or not ticker or not name:
            raise RuntimeError(f"'{sid}' 전략의 운용 계좌·종목이 없습니다 — 화면에서 골라 저장하세요.")
        try:
            strategies[sid] = {
                "account_id": account_id,
                "ticker": ticker,
                "name": name,
                **validate_strategy_trade_config({key: entry.get(key) for key in EDITABLE_CONFIG_KEYS}),
            }
        except ValueError as error:
            raise ValueError(f"'{sid}' 전략 설정이 올바르지 않습니다: {error}") from error

    return {
        "slack_enabled": bool(doc.get("slack_enabled", False)),
        "strategies": strategies,
    }


def etf_options() -> list[dict[str, str]]:
    """전략에 쓸 수 있는 종목 — kor_etf 종목풀의 ETF(티커·이름)."""
    from utils.stock_list_io import get_etfs

    options = [
        {"ticker": str(item.get("ticker") or "").strip(), "name": str(item.get("name") or "").strip()}
        for item in get_etfs(ETF_POOL) or []
    ]
    return [option for option in options if option["ticker"] and option["name"]]


def _validated_target(config: dict[str, Any], *, own_strategy_id: str | None) -> dict[str, Any]:
    """편집 입력에서 계좌·종목·파라미터를 검증해 저장 값으로 돌려준다. 실패 시 ValueError."""
    if not isinstance(config, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")
    account_id = str(config.get("account_id") or "").strip()
    if account_id not in {option["account_id"] for option in account_options()}:
        raise ValueError(f"운용 계좌로 쓸 수 없는 계좌입니다: {config.get('account_id')}")
    ticker = str(config.get("ticker") or "").strip()
    etf = next((option for option in etf_options() if option["ticker"] == ticker), None)
    if etf is None:
        raise ValueError(f"{ETF_POOL} 종목풀에 없는 종목입니다: {config.get('ticker')}")
    # 같은 계좌·같은 종목을 두 전략이 쓰면 같은 보유로 회차를 두 번 세게 된다.
    for sid, other in load_settings()["strategies"].items():
        if sid != own_strategy_id and other["account_id"] == account_id and other["ticker"] == ticker:
            raise ValueError(f"이 계좌의 {etf['name']}({ticker}) 전략이 이미 있습니다.")
    return {"account_id": account_id, "ticker": ticker, "name": etf["name"], **validate_strategy_trade_config(config)}


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
    # 계좌나 종목이 바뀌면 그 전략의 알림 중복 방지 기록을 비운다 — 이전 보유 상태로 새 신호를
    # 판정하면 엉뚱한 확인·재알림이 나간다.
    if previous is not None and (previous["account_id"], previous["ticker"]) != (
        target["account_id"],
        target["ticker"],
    ):
        update["$unset"] = _sent_log_unset(db, strategy_id)
        if not update["$unset"]:
            del update["$unset"]
    db[_CONFIG_COLLECTION].update_one({"_id": _SETTINGS_KEY}, update, upsert=True)


def _sent_log_unset(db: Any, strategy_id: str) -> dict[str, str]:
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}, {"sent_log": 1}) or {}
    return {f"sent_log.{key}": "" for key in (doc.get("sent_log") or {}) if str(key).startswith(f"{strategy_id}:")}


def save_strategy_settings(strategy_id: str, *, config: dict[str, Any]) -> dict[str, Any]:
    """전략 하나의 종목·운용 계좌·파라미터를 저장하고, 전체 설정을 반환한다."""
    sid = str(strategy_id or "").strip()
    current = load_settings()["strategies"]
    if sid not in current:
        raise ValueError(f"알 수 없는 전략입니다: {strategy_id}")
    _save_strategy(sid, _validated_target(config, own_strategy_id=sid), previous=current[sid])
    return load_settings()


def add_strategy(config: dict[str, Any]) -> dict[str, Any]:
    """전략을 하나 추가하고, 전체 설정을 반환한다."""
    from uuid import uuid4

    target = _validated_target(config, own_strategy_id=None)
    _save_strategy(f"s{uuid4().hex[:8]}", target, previous=None)
    return load_settings()


def delete_strategy(strategy_id: str) -> dict[str, Any]:
    """전략을 삭제하고(알림 중복 방지 기록 포함), 전체 설정을 반환한다."""
    from utils.db_manager import get_db_connection

    sid = str(strategy_id or "").strip()
    if sid not in load_settings()["strategies"]:
        raise ValueError(f"알 수 없는 전략입니다: {strategy_id}")
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    unset = {f"strategies.{sid}": "", **_sent_log_unset(db, sid)}
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY}, {"$set": {"updated_at": datetime.now().isoformat()}, "$unset": unset}
    )
    return load_settings()


def save_slack_enabled(enabled: bool) -> dict[str, Any]:
    """전역 슬랙 스위치를 저장하고, 전체 설정을 반환한다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {"slack_enabled": bool(enabled), "updated_at": datetime.now().isoformat()}},
        upsert=True,
    )
    return load_settings()


def account_options() -> list[dict[str, Any]]:
    """운용 계좌 선택지 — 한국 종목을 담는 계좌(국가 kor). 표기용 이름·아이콘·순서를 함께 준다."""
    from utils.settings_loader import get_account_settings, list_available_accounts

    options: list[dict[str, Any]] = []
    for account_id in list_available_accounts():
        settings = get_account_settings(account_id) or {}
        inner = settings.get("settings") or settings
        if str(inner.get("country_code") or "").strip().lower() != "kor":
            continue
        options.append(
            {
                "account_id": account_id,
                "name": inner.get("name"),
                "icon": inner.get("icon"),
                "order": inner.get("order"),
            }
        )
    return sorted(options, key=lambda item: (item["order"] is None, item["order"]))


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


def _load_unheld_closes(tickers: list[str]) -> dict[str, float]:
    """미보유 종목의 현재가 — 실시간 시세 우선, 실패분은 가격 캐시 폴백.

    1호 진입 판정에는 아직 안 산 종목의 현재가가 필요하다(없으면 매수 신호가
    영원히 안 뜬다). 실시간(pools-rank 와 같은 소스)을 1순위로 쓰고, 캐시 폴백은
    전략 종목이 든 풀(`PRICE_CACHE_TICKER_TYPES`)을 본다.
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

        frames = load_cached_frames_bulk_from_ticker_types(list(PRICE_CACHE_TICKER_TYPES), remaining)
        for ticker in remaining:
            frame = frames.get(ticker)
            if frame is None or frame.empty or "Close" not in frame.columns:
                continue
            series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if not series.empty:
                closes[ticker] = float(series.iloc[-1])
    return closes


def _round_to_tick(price: float) -> float:
    """KRX ETF 호가 단위(5원)로 반올림한다 — 지정가는 이 단위로만 주문할 수 있다."""
    return round(price / 5) * 5


def _load_ticker_status(ticker: str, name: str, trigger_pct: float, live_close: float | None) -> dict[str, Any]:
    """판정 기준 시세 — 이 전략 종목 자신의 현재가·전일 종가·1호 진입선·최근 저점.

    (이전에는 코스피/코스닥 지수를 썼지만, 이제 판정·표시 모두 자기 종목 가격이다)
    """
    import pandas as pd

    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types

    frames = load_cached_frames_bulk_from_ticker_types(list(PRICE_CACHE_TICKER_TYPES), [ticker])
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

    # 최근 3개월(92일) 저점 — 회차 지정가가 어느 층에 걸리는지 가늠하는 기준선.
    # 가격 캐시는 확정 종가라 오늘 장중 현재가가 새 저점이어도 빠져 있다 — 현재가가 더 낮으면
    # 오늘이 저점이다(그대로 두면 현재가가 3개월 저점보다 낮은 값으로 표시됐다).
    low_window = close[close.index >= close.index[-1] - pd.Timedelta(days=92)]
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
        "buy_trigger": _round_to_tick(prev_close * (1.0 - float(trigger_pct) / 100.0)),
        "recent_low": round(low_value),
        "recent_low_date": low_date.strftime("%Y-%m-%d"),
        "recent_low_label": "3개월 저점",
    }


def _build_strategy_view(
    strategy_id: str,
    strategy_settings: dict[str, Any],
    account_rows: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """전략 하나의 운용 현황(회차 표·상태)을 계좌 보유에서 역산한다."""
    ticker = str(strategy_settings["ticker"])
    name = str(strategy_settings["name"])
    trigger_pct = float(strategy_settings["trigger_pct"])
    round_quantity = int(strategy_settings["round_quantity"])
    max_rounds = int(strategy_settings["rounds"])
    down_ratio = 1.0 - trigger_pct / 100.0
    up_ratio = 1.0 + trigger_pct / 100.0

    account = account_rows.get(ticker) or {}
    quantity = int(account.get("quantity") or 0)
    avg_price = float(account.get("avg_price") or 0)
    live_close = float(account.get("close") or 0) or _load_unheld_closes([ticker]).get(ticker)
    index_status = _load_ticker_status(ticker, name, trigger_pct, live_close)
    close = float(index_status["close"])

    # ── 회차 역산 — 수량으로 진입 회차 수, 평균단가로 1호가 ──
    held_rounds = 0
    first_entry: float | None = None
    quantity_mismatch = False
    if quantity > 0 and avg_price > 0:
        held_rounds = max(1, min(max_rounds, round(quantity / round_quantity)))
        quantity_mismatch = quantity != held_rounds * round_quantity
        ladder_sum = sum(down_ratio**i for i in range(held_rounds))
        first_entry = avg_price * held_rounds / ladder_sum

    def entry_price(round_no: int) -> float | None:
        if first_entry is None:
            return None
        return first_entry * down_ratio ** (round_no - 1)

    rows: list[dict[str, Any]] = []
    for round_no in range(1, max_rounds + 1):
        held = round_no <= held_rounds
        is_next = round_no == held_rounds + 1
        entry = entry_price(round_no)
        if held and entry is not None:
            sell_limit = _round_to_tick(entry * up_ratio)
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
                    "is_next": False,
                }
            )
            continue
        # 미보유 회차 — 사다리 자리(보유가 있으면 1호가 기준, 없으면 전일 종가 기준).
        # 지정가로 바로 쓸 수 있게 호가 단위로 반올림한다.
        if first_entry is not None:
            buy_limit = first_entry * down_ratio ** (round_no - 1)
        else:
            buy_limit = float(index_status["prev_close"]) * down_ratio**round_no
        buy_limit = _round_to_tick(buy_limit)
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
                "buy_reached": bool(is_next and close > 0 and close <= buy_limit),
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
        "ticker": ticker,
        "account_id": strategy_settings["account_id"],
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
        },
        "rounds": rows,
    }


def load_strategy_trade_view() -> dict[str, Any]:
    """전 전략의 운용 현황을 화면용으로 묶어 반환한다."""
    settings = load_settings()
    # 계좌 조회는 계좌당 1회 — 같은 계좌를 쓰는 전략끼리 나눠 쓴다.
    rows_by_account = {
        account_id: _load_account_rows(account_id)
        for account_id in {entry["account_id"] for entry in settings["strategies"].values()}
    }
    return {
        "accounts": account_options(),
        "etfs": etf_options(),
        "slack_enabled": settings["slack_enabled"],
        "strategies": [
            _build_strategy_view(sid, entry, rows_by_account[entry["account_id"]])
            for sid, entry in settings["strategies"].items()
        ],
    }
