"""전략 사고팔기 운용 현황 조회 서비스 (UI/API 용).

전략 2개(코스피200 / 코스닥150)를 같은 계좌에서 독립 운용한다. 전략별 파라미터(%)는
DB(``system_config.strategy_trade_settings.strategies``)가 단일 소스이며 화면에서
편집한다. 회차 티커는 전략별 코드 고정이고, 슬랙 스위치는 전역 1개다.

보유 현황은 백테스트가 아니라 ``ACCOUNT_ID`` 계좌의 **실제 보유 수량·평균단가**를
읽어 계산한다. 배치는 없다 — 화면을 열 때마다 즉시 계산한다.

회차 부여 규칙 (전략별 동일)
---------------------------
- 보유 종목은 **평균단가 내림차순**으로 1호부터 다시 번호를 매긴다.
- 매도 지정가는 각 종목의 자기 평균단가 ``+take_profit_pct``.
- 다음 매수는 **가장 마지막에 매수한 가격(= 보유 중 최저 평균단가)** 의
  ``-add_drop_pct`` 이며, 종목은 미보유 중 고정 회차가 가장 빠른 것을 쓴다.
  (회차 종목이 모두 같은 지수를 추종해 가격 차이가 작다는 전제)
- 보유가 하나도 없으면 1호 진입 대기 상태다 — 판정 지수가 ``entry_drop_pct``
  이상 하락한 날 고정 1호 종목을 매수한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.strategy_trade_config import (
    ACCOUNT_ID,
    EDITABLE_PCT_KEYS,
    PRICE_CACHE_TICKER_TYPE,
    STRATEGIES,
    STRATEGY_IDS,
    validate_strategy_trade_config,
)

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def load_settings() -> dict[str, Any]:
    """저장된 슬랙 스위치 + 전략별 파라미터(%)를 반환한다.

    전략 항목이 없거나 깨졌으면 시드값으로 슬쩍 넘어가지 않고 에러를 낸다 —
    그럴듯한 값이 화면에 떴다가 그대로 저장돼 실제 설정이 덮어써지는 것을 막는다.

    반환: ``{"slack_enabled", "strategies": {sid: {"trigger_pct": float}}}``
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
    for sid in STRATEGY_IDS:
        entry = stored.get(sid)
        if not isinstance(entry, dict):
            raise RuntimeError(f"'{sid}' 전략 설정이 없습니다 — 마이그레이션/저장이 필요합니다.")
        try:
            strategies[sid] = validate_strategy_trade_config(
                {key: entry.get(key) for key in EDITABLE_PCT_KEYS}
            )
        except ValueError as error:
            raise ValueError(f"'{sid}' 전략 설정이 올바르지 않습니다: {error}") from error

    return {"slack_enabled": bool(doc.get("slack_enabled", False)), "strategies": strategies}


def save_strategy_settings(strategy_id: str, *, config: dict[str, Any]) -> dict[str, Any]:
    """전략 하나의 파라미터(%)를 저장하고, 전체 설정을 반환한다."""
    from utils.db_manager import get_db_connection

    sid = str(strategy_id or "").strip()
    if sid not in STRATEGY_IDS:
        raise ValueError(f"알 수 없는 전략입니다: {strategy_id}")

    updates: dict[str, Any] = {"updated_at": datetime.now().isoformat()}
    for key, value in validate_strategy_trade_config(config).items():
        updates[f"strategies.{sid}.{key}"] = value

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    db[_CONFIG_COLLECTION].update_one({"_id": _SETTINGS_KEY}, {"$set": updates}, upsert=True)
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


def _load_account_rows() -> dict[str, dict[str, float]]:
    """계좌에서 보유 종목의 수량·평균단가·현재가를 읽는다 (두 전략 공용, 1회 조회).

    ``load_real_holdings_table`` 은 실시간 시세를 반영한 현재가까지 채워준다.
    """
    from utils.portfolio_io import load_real_holdings_table

    frame = load_real_holdings_table(ACCOUNT_ID)
    if frame is None or frame.empty:
        raise RuntimeError(f"{ACCOUNT_ID} 보유 종목을 불러올 수 없습니다.")

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
    """미보유 회차 종목의 현재가 — 실시간 시세 우선, 실패분은 가격 캐시 폴백.

    다음 회차 매수 판정에는 아직 안 산 종목의 현재가가 필요하다(없으면 매수 신호가
    영원히 안 뜬다). 코스닥150 회차 종목은 종목풀(가격 캐시) 밖에 있는 것이 많아
    실시간(pools-rank 와 같은 소스)을 1순위로 쓴다.
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
        from utils.cache_utils import load_cached_frames_bulk_from_ticker_types

        import pandas as pd

        frames = load_cached_frames_bulk_from_ticker_types([PRICE_CACHE_TICKER_TYPE], remaining)
        for ticker in remaining:
            frame = frames.get(ticker)
            if frame is None or frame.empty or "Close" not in frame.columns:
                continue
            series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if not series.empty:
                closes[ticker] = float(series.iloc[-1])
    return closes


def _index_level_for(
    limit_price: float | None, close: float | None, index_close: float
) -> float | None:
    """지정가에 닿을 때의 대략적인 지수 수준.

    회차 ETF 가격은 추종 지수에 비례하므로 ``지수 x (지정가 / 현재가)`` 로 환산한다.
    추종 지수(코스피200/코스닥150)와 표시 지수(코스피/코스닥)가 달라 정확한 값은
    아니다 — 참고용.
    """
    if limit_price is None or close is None or close <= 0:
        return None
    return round(index_close * (limit_price / close), 2)


def _load_index_status(index_ticker: str, index_name: str, trigger_pct: float) -> dict[str, Any]:
    """1호 진입 판정용 지수 현황 — 최근 종가와 진입 트리거 가격."""
    from utils.market_trend_service import load_index_ohlc

    frame = load_index_ohlc(index_ticker)
    if frame is None or frame.empty:
        raise RuntimeError(f"{index_name} 지수 데이터를 불러올 수 없습니다.")

    import pandas as pd

    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        raise RuntimeError(f"{index_name} 지수 종가가 없습니다.")

    entry_ratio = 1.0 - float(trigger_pct) / 100.0
    last_close = float(close.iloc[-1])

    # 최근 3개월(92일) 저점 — 회차 지정가가 지수 어느 층에 걸리는지 가늠하는 기준선.
    low_window = close[close.index >= close.index[-1] - pd.Timedelta(days=92)]
    low_date = low_window.idxmin()

    return {
        "name": index_name,
        "as_of": close.index[-1].strftime("%Y-%m-%d"),
        "close": round(last_close, 2),
        "buy_trigger": round(last_close * entry_ratio, 2),
        "recent_low": round(float(low_window.min()), 2),
        "recent_low_date": low_date.strftime("%Y-%m-%d"),
        "recent_low_label": "3개월 저점",
    }


def _build_strategy_view(
    strategy_id: str,
    strategy_settings: dict[str, Any],
    account_rows: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """전략 하나의 운용 현황(회차 표·상태)을 계산한다."""
    meta = STRATEGIES[strategy_id]
    names = {code: name for code, name in meta["round_tickers"]}
    tickers = [code for code, _ in meta["round_tickers"]]

    unheld_tickers = [
        t for t in tickers if not ((account_rows.get(t) or {}).get("quantity") or 0)
    ]
    unheld_closes = _load_unheld_closes(unheld_tickers)
    trigger_pct = float(strategy_settings["trigger_pct"])
    index_status = _load_index_status(meta["index_ticker"], meta["index_name"], trigger_pct)

    take_profit = 1.0 + trigger_pct / 100.0
    add_ratio = 1.0 - trigger_pct / 100.0

    # 계좌에 없는 티커는 미보유로 다룬다(0주로 등록된 경우도 동일).
    held: list[dict[str, Any]] = []
    unheld: list[dict[str, Any]] = []
    for fixed_round, ticker in enumerate(tickers, start=1):
        row = account_rows.get(ticker) or {}
        quantity = int(row.get("quantity") or 0)
        avg_price = float(row.get("avg_price") or 0)
        # 보유분은 실시간이 반영된 보유 테이블 값을, 미보유분은 실시간/캐시 값을 쓴다.
        close = float(row.get("close") or 0) or float(unheld_closes.get(ticker) or 0)
        entry = {
            "ticker": ticker,
            "name": names.get(ticker, ticker),
            "fixed_round": fixed_round,
            "avg_price": avg_price if avg_price > 0 else None,
            "close": round(close, 2) if close > 0 else None,
        }
        if quantity > 0 and avg_price > 0:
            held.append(entry)
        else:
            unheld.append(entry)

    # 보유분은 평균단가 내림차순 — 비싸게 잡힌 것이 1호다.
    held.sort(key=lambda item: item["avg_price"], reverse=True)
    # 가장 마지막에 매수한 가격 = 보유 중 최저 평균단가.
    last_buy_price = held[-1]["avg_price"] if held else None
    # 다음 매수 종목은 미보유 중 고정 회차가 가장 빠른 것.
    unheld.sort(key=lambda item: item["fixed_round"])
    next_target = unheld[0] if unheld else None

    rows: list[dict[str, Any]] = []
    for display_round, item in enumerate(held, start=1):
        avg_price = float(item["avg_price"])
        close = item["close"]
        sell_limit = avg_price * take_profit
        rows.append(
            {
                **item,
                "round": display_round,
                "held": True,
                "profit_pct": (
                    round((close / avg_price - 1.0) * 100.0, 2)
                    if close is not None and avg_price > 0
                    else None
                ),
                "avg_price": round(avg_price, 2),
                "sell_limit": round(sell_limit, 2),
                "sell_index": _index_level_for(sell_limit, close, index_status["close"]),
                "sell_reached": bool(close is not None and close >= sell_limit),
                "buy_limit": None,
                # 이미 매수한 회차는 평균단가가 어느 지수 수준이었는지 환산해 보여준다.
                "buy_index": _index_level_for(avg_price, close, index_status["close"]),
                "buy_reached": False,
                "is_next": False,
            }
        )

    entry_ratio = 1.0 - trigger_pct / 100.0
    # 하락 판정은 **마지막 매수 종목**의 가격으로 한다(규칙: 직전 회차 종목이 자기
    # 진입가 대비 -간격% 하락하면 다음 회차 매수). 같은 지수 추종이라도 운용사마다
    # 가격 스케일이 다르므로(예: TIGER 코스닥150 이 KODEX 보다 +2%), 체인 지정가를
    # 다른 종목 가격에 그대로 대면 필요 하락률이 왜곡된다 — 각 회차의 표시 지정가는
    # '필요 하락 비율'을 자기 현재가에 곱해 자기 스케일로 환산한다.
    monitored_close = held[-1]["close"] if held else None
    for offset, item in enumerate(unheld):
        is_next = next_target is not None and item["ticker"] == next_target["ticker"]
        close = item["close"]
        if last_buy_price is not None:
            # 감시 종목 기준 체인 레벨 → 지금부터 필요한 하락 비율 → 자기 스케일 지정가.
            chain_level = last_buy_price * (add_ratio ** (offset + 1))
            required_ratio = (chain_level / monitored_close) if monitored_close else None
            buy_limit = close * required_ratio if (close and required_ratio is not None) else None
        else:
            # 보유가 없으면 1호는 자기 현재가 -간격% 가 기준(자기 스케일이라 왜곡 없음).
            buy_limit = close * entry_ratio * (add_ratio**offset) if close else None
        rows.append(
            {
                **item,
                "round": len(held) + offset + 1,
                "held": False,
                "profit_pct": None,
                "sell_limit": None,
                "sell_index": None,
                "sell_reached": False,
                "buy_limit": round(buy_limit, 2) if buy_limit is not None else None,
                "buy_index": _index_level_for(buy_limit, close, index_status["close"]),
                "buy_reached": bool(
                    buy_limit is not None and close is not None and close <= buy_limit
                ),
                "is_next": is_next,
            }
        )

    # 다음 회차 매수까지 남은 하락률(%) — 다음 대상 회차의 지정가 ÷ 현재가.
    # ETF 가격과 지수는 비례하므로 '지수가 이만큼 내리면 산다'와 같은 값이다.
    # 이미 도달했으면 0 이상(+)이 나온다. 다음 대상이 없거나 가격이 없으면 None.
    next_buy_drop_pct = None
    next_row = next((r for r in rows if r["is_next"]), None)
    if next_row is not None and next_row["buy_limit"] is not None and next_row["close"]:
        next_buy_drop_pct = round((next_row["buy_limit"] / next_row["close"] - 1.0) * 100.0, 2)

    # 가장 가까운 매도 회차 — 보유 중 목표가까지 남은 상승률이 최소인 회차.
    # 값이 0 이하면 이미 도달. 보유가 없으면 None.
    next_sell_round = None
    next_sell_rise_pct = None
    sell_candidates = [
        (r["round"], (r["sell_limit"] / r["close"] - 1.0) * 100.0)
        for r in rows
        if r["held"] and r["sell_limit"] is not None and r["close"]
    ]
    if sell_candidates:
        next_sell_round, rise = min(sell_candidates, key=lambda pair: pair[1])
        next_sell_rise_pct = round(rise, 2)

    return {
        "strategy_id": strategy_id,
        "label": meta["label"],
        "config": {
            "trigger_pct": trigger_pct,
            "rounds": len(tickers),
            "index_name": meta["index_name"],
        },
        "index": index_status,
        "status": {
            "held_count": len(held),
            "waiting_first_entry": not held,
            "next_round": (len(held) + 1) if next_target is not None else None,
            "next_ticker": next_target["ticker"] if next_target else None,
            "next_name": next_target["name"] if next_target else None,
            "last_buy_price": round(last_buy_price, 2) if last_buy_price is not None else None,
            "next_buy_drop_pct": next_buy_drop_pct,
            "next_sell_round": next_sell_round,
            "next_sell_rise_pct": next_sell_rise_pct,
        },
        "rounds": rows,
    }


def load_strategy_trade_view() -> dict[str, Any]:
    """두 전략의 운용 현황을 화면용으로 묶어 반환한다."""
    settings = load_settings()
    account_rows = _load_account_rows()
    return {
        "account_id": ACCOUNT_ID,
        "slack_enabled": settings["slack_enabled"],
        "strategies": [
            _build_strategy_view(sid, settings["strategies"][sid], account_rows) for sid in STRATEGY_IDS
        ],
    }
