"""전략 사고팔기 운용 현황 조회 서비스 (UI/API 용).

전략 파라미터(진입 −5% / 추가 −5% / 목표 +5% / 6회차 / 티커 6종)는
``utils/strategy_trade_config.py`` 에 고정돼 있어 화면에서 편집하지 않는다.
화면에서 저장하는 값은 **슬랙 알림 스위치** 하나다.

보유 현황은 백테스트가 아니라 ``ACCOUNT_ID`` 계좌의 **실제 보유 수량·평균단가**를
읽어 계산한다. 배치는 없다 — 화면을 열 때마다 즉시 계산한다.

회차 부여 규칙
-------------
- 보유 종목은 **평균단가 내림차순**으로 1호부터 다시 번호를 매긴다. 실제 매수가
  전략 순서와 어긋나도(예: 2호가 1호보다 비싸게 잡힘) 화면은 비싼 것부터 1호로
  보여준다.
- 매도 지정가는 각 종목의 자기 평균단가 ``+take_profit_pct``.
- 다음 매수는 **가장 마지막에 매수한 가격(= 보유 중 최저 평균단가)** 의
  ``-add_drop_pct`` 이며, 종목은 미보유 중 고정 회차가 가장 빠른 것을 쓴다.
  (6종 모두 코스피200 추종이라 가격 차이가 작다는 전제)
- 보유가 하나도 없으면 1호 진입 대기 상태다 — 지수가 ``entry_drop_pct`` 이상
  하락한 날 고정 1호 종목을 매수한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.strategy_trade_config import (
    ACCOUNT_ID,
    DEFAULT_CONFIG,
    INDEX_NAME,
    INDEX_TICKER,
    ROUND_TICKERS,
    round_ticker_names,
)

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def load_settings() -> dict[str, Any]:
    """저장된 슬랙 스위치를 반환한다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return {"slack_enabled": False}
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    return {"slack_enabled": bool(doc.get("slack_enabled", False))}


def save_settings(*, slack_enabled: bool) -> dict[str, Any]:
    """슬랙 스위치를 저장하고, 저장된 설정을 반환한다."""
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {"slack_enabled": bool(slack_enabled), "updated_at": datetime.now().isoformat()}},
        upsert=True,
    )
    return load_settings()


def _load_account_rows() -> dict[str, dict[str, float]]:
    """계좌에서 회차 종목의 수량·평균단가·현재가를 읽는다.

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


def _index_level_for(
    limit_price: float | None, close: float | None, index_close: float
) -> float | None:
    """지정가에 닿을 때의 대략적인 코스피 지수.

    코스피200 ETF 가격은 지수에 비례하므로 ``지수 x (지정가 / 현재가)`` 로 환산한다.
    추종 대상이 코스피200 이고 표시 지수는 코스피라 정확한 값은 아니다 — 참고용.
    """
    if limit_price is None or close is None or close <= 0:
        return None
    return round(index_close * (limit_price / close), 2)


def _load_index_status() -> dict[str, Any]:
    """1호 진입 판정용 지수 현황 — 최근 종가와 진입 트리거 가격."""
    from utils.market_trend_service import load_index_ohlc

    frame = load_index_ohlc(INDEX_TICKER)
    if frame is None or frame.empty:
        raise RuntimeError(f"{INDEX_NAME} 지수 데이터를 불러올 수 없습니다.")

    import pandas as pd

    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        raise RuntimeError(f"{INDEX_NAME} 지수 종가가 없습니다.")

    entry_ratio = 1.0 - float(DEFAULT_CONFIG["entry_drop_pct"]) / 100.0
    last_close = float(close.iloc[-1])
    return {
        "name": INDEX_NAME,
        "as_of": close.index[-1].strftime("%Y-%m-%d"),
        "close": round(last_close, 2),
        "buy_trigger": round(last_close * entry_ratio, 2),
    }


def load_strategy_trade_view() -> dict[str, Any]:
    """계좌 실제 보유 기준의 운용 현황을 화면용으로 묶어 반환한다."""
    settings = load_settings()
    rounds_count = int(DEFAULT_CONFIG["rounds"])

    names = round_ticker_names()
    tickers = [code for code, _ in ROUND_TICKERS]
    account_rows = _load_account_rows()
    index_status = _load_index_status()

    take_profit = 1.0 + float(DEFAULT_CONFIG["take_profit_pct"]) / 100.0
    add_ratio = 1.0 - float(DEFAULT_CONFIG["add_drop_pct"]) / 100.0

    # 계좌에 없는 티커는 미보유로 다룬다(0주로 등록된 경우도 동일).
    held: list[dict[str, Any]] = []
    unheld: list[dict[str, Any]] = []
    for fixed_round, ticker in enumerate(tickers, start=1):
        row = account_rows.get(ticker) or {}
        quantity = int(row.get("quantity") or 0)
        avg_price = float(row.get("avg_price") or 0)
        close = float(row.get("close") or 0)
        entry = {
            "ticker": ticker,
            "name": names.get(ticker, ticker),
            "fixed_round": fixed_round,
            "avg_price": avg_price if avg_price > 0 else None,
            "close": round(close, 2) if close > 0 else None,
            # 평가금액 합계에만 쓰는 내부 값 — 화면 그리드에는 노출하지 않는다.
            "_quantity": quantity,
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
        item = {key: value for key, value in item.items() if key != "_quantity"}
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

    entry_ratio = 1.0 - float(DEFAULT_CONFIG["entry_drop_pct"]) / 100.0
    for offset, item in enumerate(unheld):
        is_next = next_target is not None and item["ticker"] == next_target["ticker"]
        close = item["close"]
        item = {key: value for key, value in item.items() if key != "_quantity"}
        # 미보유 회차의 지정가는 마지막 매수가에서 회차마다 add_drop_pct 씩 낮춰
        # 연쇄로 잡는다(다음 회차뿐 아니라 그 이후 회차까지 미리 보여준다).
        # 보유가 없으면 1호는 현재가 -entry_drop_pct 가 기준이 된다.
        base_price = last_buy_price if last_buy_price is not None else (close * entry_ratio if close else None)
        exponent = offset + 1 if last_buy_price is not None else offset
        buy_limit = base_price * (add_ratio**exponent) if base_price is not None else None
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

    holdings_value = sum(
        float(item["_quantity"]) * float(item["close"])
        for item in held
        if item["close"] is not None
    )
    invested = sum(float(item["_quantity"]) * float(item["avg_price"]) for item in held)

    return {
        "account_id": ACCOUNT_ID,
        "config": {
            "entry_drop_pct": DEFAULT_CONFIG["entry_drop_pct"],
            "add_drop_pct": DEFAULT_CONFIG["add_drop_pct"],
            "take_profit_pct": DEFAULT_CONFIG["take_profit_pct"],
            "rounds": rounds_count,
            "index_name": INDEX_NAME,
        },
        "slack_enabled": settings["slack_enabled"],
        "index": index_status,
        "status": {
            "held_count": len(held),
            "waiting_first_entry": not held,
            "next_round": (len(held) + 1) if next_target is not None else None,
            "next_ticker": next_target["ticker"] if next_target else None,
            "next_name": next_target["name"] if next_target else None,
            "last_buy_price": round(last_buy_price, 2) if last_buy_price is not None else None,
            "invested_amount": round(invested),
            "valuation_amount": round(holdings_value),
            "profit_amount": round(holdings_value - invested) if held else None,
            "profit_pct": (
                round((holdings_value / invested - 1.0) * 100.0, 2) if invested > 0 else None
            ),
        },
        "rounds": rows,
    }
