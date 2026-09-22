"""확정 일봉에 실시간 마지막 봉을 붙이는 **단일 규칙** — 순위·전략·알람이 공유한다.

예전에는 이 일을 두 곳이 서로 다르게 했다. 순위(`utils.rankings`)는 마지막 봉을 항상
실시간으로 갈아끼웠고, 전략(`utils.slot_positions` + `core.strategy.intraday`)은
「마지막 체결일 > 캐시 마지막 봉 날짜」일 때만 붙였다. 가격 캐시는 장중에도 그날 봉을
쓰기 때문에(증분 배치는 이미 있는 날짜를 다시 안 가져온다) 전략만 장중 스냅샷으로
판정했고, 같은 시각에 두 화면의 진입 목록이 갈렸다. 규칙을 여기 한 곳에 둔다.

규칙
- 실시간 가격은 **마지막 봉을 교체**한다.
- 그날 정규장이 이미 시작됐는데 캐시에 그 날짜 봉이 없으면 그 날짜로 **새 봉을 추가**한다.
  장 시작 전(프리·데이장)과 휴장일에는 새 봉을 만들지 않는다 — 가짜 봉을 만들지 않는다.
- 실시간이 없는 종목은 확정 봉을 **그대로 둔다**. 다만 새로 추가한 봉에는 값이 없어
  NaN 으로 남고 판정 불가가 된다(strategy_logic.md 「장중 잠정 실행」 — 값을 지어내지 않는다).

어느 세션의 가격을 줄지는 시세 소스(`utils.realtime_quotes`)가 정한다. 지금 열려 있는
세션(프리·정규·애프터·데이장)이 있으면 그 세션 가격을, 모두 닫혔으면 정규장 종가를
`nowVal` 로 준다 — 애프터 가격으로 마감하지 않고 정규장 종가로 되돌아온다.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pandas as pd

from utils.market_session import market_today, regular_session_started


def _normalize_day(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def effective_bar_date(
    cached_last: Any,
    country_code: str,
    *,
    now: datetime | None = None,
) -> pd.Timestamp:
    """실시간 가격을 붙일 봉의 날짜 — 마지막 봉, 또는 오늘 정규장이 열렸으면 오늘."""
    last_day = _normalize_day(cached_last)
    if not regular_session_started(country_code, now=now):
        return last_day
    today = _normalize_day(market_today(country_code, now=now))
    return today if today > last_day else last_day


def _realtime_price(entry: Mapping[str, Any] | None) -> float | None:
    """시세 항목의 실시간 가격. 소스가 이미 현재 세션에 맞는 값을 담아 준다."""
    if not isinstance(entry, Mapping) or not entry:
        return None
    raw = entry.get("nowVal")
    if raw is None:
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def apply_realtime_close(
    cached_close_series: pd.Series | None,
    realtime_entry: Mapping[str, Any] | None,
    country_code: str,
    *,
    now: datetime | None = None,
) -> pd.Series | None:
    """확정 종가 시리즈에 실시간 마지막 봉을 반영한다(한 종목)."""
    if cached_close_series is None or cached_close_series.empty:
        return None

    price = _realtime_price(realtime_entry)
    if price is None:
        return cached_close_series

    adjusted = cached_close_series.copy()
    adjusted.index = pd.DatetimeIndex([_normalize_day(idx) for idx in adjusted.index])
    target = effective_bar_date(adjusted.index[-1], country_code, now=now)
    adjusted.loc[target] = price
    return adjusted.sort_index()


def apply_realtime_closes(
    frame: pd.DataFrame,
    live_prices: Mapping[str, float],
    bar_date: Any,
) -> pd.DataFrame:
    """확정 (날짜 × 티커) 프레임에 실시간 마지막 봉을 반영한다(여러 종목).

    붙일 봉의 날짜는 **인자로 받는다**. 한 실행 안에서 종가·고가·시가 프레임을 각각
    부르는데 함수가 매번 날짜를 다시 재면 세션 경계에서 프레임끼리 어긋난다 —
    호출부가 `effective_bar_date` 로 한 번 정한 값을 그대로 넘긴다.

    기존 봉을 교체할 때는 실시간이 있는 종목만 덮어쓴다 — 행을 통째로 갈아끼우면
    실시간이 없는 종목의 확정 종가가 NaN 으로 날아간다.
    """
    if frame is None or frame.empty:
        return frame

    effective = frame.copy()
    effective.index = pd.DatetimeIndex([_normalize_day(idx) for idx in effective.index])

    values = {
        ticker: float(price)
        for ticker, price in (live_prices or {}).items()
        if ticker in effective.columns and price is not None and float(price) > 0
    }

    target = _normalize_day(bar_date)
    if target not in effective.index:
        # 새 봉 — 실시간이 없는 종목은 값이 없으니 NaN 으로 둔다(판정 불가).
        effective.loc[target] = pd.Series(values).reindex(effective.columns)
        return effective.sort_index()

    for ticker, price in values.items():
        effective.loc[target, ticker] = price
    return effective
