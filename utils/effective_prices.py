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


def quote_trade_day(entry: Mapping[str, Any] | None) -> pd.Timestamp | None:
    """시세 항목의 **마지막 체결 날짜**(시장 현지). 소스가 시각을 안 주면 None.

    미국은 `localTradedAt`(토스 `tradeDateTime` 을 ET 로 변환한 값), 한국 개별주는
    네이버 `localTradedAt` 이다. 한국 ETF·호주·해외 지연 시세는 시각을 주지 않아 None 이고,
    그때는 신선도를 **검증할 수 없다**(추정하지 않는다).
    """
    if not isinstance(entry, Mapping) or not entry:
        return None
    raw = entry.get("localTradedAt") or entry.get("tradeDateTime")
    if not raw:
        return None
    try:
        stamp = pd.Timestamp(raw)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return _normalize_day(stamp)


def last_regular_close(entry: Mapping[str, Any] | None) -> float | None:
    """시세가 알려 주는 **마지막으로 마감된 정규장 종가**. 모르면 None.

    앵커 날짜(`bar_anchor`) 봉의 확정 종가가 필요한 자리에 쓴다. 가격 캐시의 그 봉을 그냥
    읽으면 안 된다 — 증분 배치가 장중에 그날 봉을 쓰고 실시간이 그 봉을 교체하므로,
    캐시의 마지막 봉은 **설계상 잠정값**이다. 미국은 정규장 마감(05:00 KST) 후 전체
    재정렬(17:10 KST)까지 12시간 동안 그 잠정 봉이 남는다 — 실측(2026-09-24 데이장)에서
    us_stock 20종목 전부 캐시 09-23 봉이 실제 종가와 달랐다(MRNA 184.27 vs 182.11).

    `prevClose` 와 구분한다. 토스는 세션을 따라가지만(애프터·데이장이면 그날 종가)
    거래소 시세 API 의 `prevClose` 는 날짜 기준 전일 종가여서 마감 후에는 하루 어긋난다.
    그래서 **의미를 실측으로 확정한 소스만** 이 키를 채운다(지금은 토스 미국).
    """
    if not isinstance(entry, Mapping) or not entry:
        return None
    raw = entry.get("lastRegularClose")
    if raw is None:
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def confirmed_close_before(close_series: pd.Series | None, bar_date: Any) -> float | None:
    """`bar_date` 보다 **앞선** 마지막 확정 종가. 없으면 None.

    앵커 봉(`bar_anchor`)의 전일 대비 변동을 재는 분모다. 앵커 봉 자체는 잠정값일 수
    있지만(`last_regular_close` 주석) 그 앞 봉은 이미 전체 재정렬을 거친 확정 종가다.
    """
    if close_series is None or close_series.empty:
        return None
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return None
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    position = int(index.normalize().searchsorted(_normalize_day(bar_date), side="left")) - 1
    if position < 0:
        return None
    value = float(series.iloc[position])
    return value if value > 0 else None


def _realtime_price(entry: Mapping[str, Any] | None, bar_day: pd.Timestamp) -> float | None:
    """붙여도 되는 실시간 가격. 소스가 이미 현재 세션에 맞는 값을 담아 준다.

    **붙일 봉보다 오래된 시세는 버린다.** 시세 소스가 멈춰 있거나(그 세션에 체결이 없는
    종목) 캐시된 값을 주면 며칠 전 가격이 오늘 봉으로 들어간다 — 재현: 금요일 시세 99.0 이
    화요일 봉에 그대로 들어갔다. 거래 시각을 모르는 소스는 검증을 건너뛴다(위 함수 주석).
    """
    if not isinstance(entry, Mapping) or not entry:
        return None
    raw = entry.get("nowVal")
    if raw is None:
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    trade_day = quote_trade_day(entry)
    if trade_day is not None and trade_day < bar_day:
        return None
    return price


def bar_anchor(country_code: str, *, now: datetime | None = None) -> pd.Timestamp:
    """실시간을 붙일 봉의 **기준 날짜** — 그 시장에서 마지막으로 정규장이 마감된 거래일.

    순위·합성·보유 알림이 모두 이 한 값을 쓴다. 경로마다 「가진 데이터의 마지막 날짜」로
    각자 정하면(순위는 풀 전체, 합성·알람은 종목별) 캐시가 종목마다 다른 날짜에서 끝날 때
    같은 가격이 서로 다른 봉에 들어간다 — 장 시작 전 재현: 순위는 9/21 에 새 봉을 추가하고
    합성·알람은 9/18 을 덮어썼다.

    달력 계산이라 가격 조회가 없고, **어떤 티커를 로드했는지에 의존하지 않아** 경로 간
    일치가 구조적으로 보장된다. 이 값이 결과를 가르는 구간은 장 시작 전뿐이다 —
    정규장이 시작되면 `effective_bar_date` 가 앵커와 무관하게 오늘을 반환한다.
    """
    from utils.market_session import last_closed_session_date

    return _normalize_day(last_closed_session_date(country_code, now=now))


def apply_realtime_close(
    cached_close_series: pd.Series | None,
    realtime_entry: Mapping[str, Any] | None,
    country_code: str,
    *,
    last_bar: pd.Timestamp,
    now: datetime | None = None,
) -> pd.Series | None:
    """확정 종가 시리즈에 실시간 마지막 봉을 반영한다(한 종목).

    ``last_bar`` 는 호출부가 `bar_anchor`로 정한 공통 기준 날짜다. 종목별 캐시 마지막
    날짜로 대체하면 화면마다 다른 봉에 가격을 넣게 되므로 반드시 전달해야 한다.
    """
    if last_bar is None or pd.isna(last_bar):
        raise ValueError("실시간 봉의 공통 기준 날짜(last_bar)가 필요합니다.")
    if cached_close_series is None or cached_close_series.empty:
        return None

    adjusted = cached_close_series.copy()
    adjusted.index = pd.DatetimeIndex([_normalize_day(idx) for idx in adjusted.index])
    reference = _normalize_day(last_bar)
    target = effective_bar_date(reference, country_code, now=now)

    price = _realtime_price(realtime_entry, target)
    if price is None:
        return cached_close_series

    adjusted.loc[target] = price
    return adjusted.sort_index()


def live_prices_for_bar(
    snapshot: Mapping[str, Mapping[str, Any]] | None,
    bar_date: Any,
    *,
    field: str = "nowVal",
) -> dict[str, float]:
    """스냅샷에서 **그 봉에 붙여도 되는** 값만 골라 `{티커: 값}` 으로 만든다.

    프레임 경로(`apply_realtime_closes`)는 이미 추출된 숫자를 받으므로, 신선도 검증은
    시세 항목이 남아 있는 이 자리에서 해야 한다 — 순위(단일 종목)와 전략(프레임)이
    같은 기준을 쓰도록 여기 한 곳에 둔다.

    `field` 는 종가 외 프레임(시가 등)을 만들 때 바꿔 쓴다.
    """
    target = _normalize_day(bar_date)
    live: dict[str, float] = {}
    for ticker, entry in (snapshot or {}).items():
        if not isinstance(entry, Mapping):
            continue
        if field == "nowVal":
            value = _realtime_price(entry, target)
        else:
            raw = entry.get(field)
            trade_day = quote_trade_day(entry)
            if trade_day is not None and trade_day < target:
                continue
            try:
                value = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                value = None
            if value is not None and value <= 0:
                value = None
        if value is not None:
            live[str(ticker)] = value
    return live


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
