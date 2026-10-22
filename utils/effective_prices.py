"""확정 일봉에 실시간 마지막 봉을 붙이는 **단일 규칙** — 순위·전략·알람이 공유한다.

예전에는 이 일을 두 곳이 서로 다르게 했다. 순위(`utils.rankings`)는 마지막 봉을 항상
실시간으로 갈아끼웠고, 전략(`utils.slot_positions` + `core.strategy.intraday`)은
「마지막 체결일 > 캐시 마지막 봉 날짜」일 때만 붙였다. 가격 캐시는 장중에도 그날 봉을
쓰기 때문에(증분 배치는 이미 있는 날짜를 다시 안 가져온다) 전략만 장중 스냅샷으로
판정했고, 같은 시각에 두 화면의 진입 목록이 갈렸다. 규칙을 여기 한 곳에 둔다.

규칙 — 봉은 **거래일** 단위이고, 데이장·프리장은 N일이 아니라 **N+1 거래일**의 세션이다.
- 정규장 중: 오늘 봉(없으면 새로 추가)에 현재가를 넣는다.
- 애프터·마감·휴장: 마지막 마감 봉(N)은 **정규장 종가로 확정**이다. 시간외 가격은 봉에 넣지
  않는다(표시용 현재가에만 쓴다) — 종가 확정 뒤 시간외 가격으로 판정이 뒤집히면 안 된다.
- 프리장·데이장: N+1 거래일 봉을 **새로 만들어** 그 시점 가격을 넣는다. 정규장이 열리면 같은
  봉이 이어진다. 휴장일 앞 금요일 밤 같이 다음 거래일이 없는 구간은 새 봉이 없다.
- 실시간이 없는 종목은 확정 봉을 **그대로 둔다**. 다만 새로 추가한 봉에는 값이 없어
  NaN 으로 남고 판정 불가가 된다(strategy_logic.md 「장중 잠정 실행」 — 값을 지어내지 않는다).

수집 어댑터가 세션별 원본 가격을 해석하고, 공통 시세 서비스가 마감 시세를 정규장
종가로 확정한 뒤 `nowVal` 로 전달한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from utils.market_session import (
    AFTERMARKET,
    CLOSED,
    REGULAR,
    aftermarket_in_daily_change,
    market_session,
    market_today,
    pre_open_trading_day,
    regular_session_started,
)


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
    """실시간 가격을 붙일 봉의 날짜 — 마지막 봉, 오늘 정규장이 열렸으면 오늘, 프리장·데이장이면 그 다음 거래일."""
    last_day = _normalize_day(cached_last)
    pre_open_day = pre_open_trading_day(country_code, now=now)
    if pre_open_day is not None:
        return max(_normalize_day(pre_open_day), last_day)
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


def realtime_missing(entry: Mapping[str, Any] | None) -> bool:
    """실시간 시세를 못 받은 종목인지 — 시세 항목이 없거나 현재가가 비었다.

    이때 일간(%)은 **비워 둔다**. 가격 캐시의 마지막 봉으로 계산하면 지난 거래일 변동이 오늘 일간처럼
    보여 전거래일(%)과 같은 값이 나란히 찍혔다(2026-10-22 토스 API 장애). 현재가는 마지막 종가를 두되
    화면이 「실시간 아님」을 알린다(`realtime_missing_tickers`).
    """
    return not isinstance(entry, Mapping) or entry.get("nowVal") is None


def last_regular_close(entry: Mapping[str, Any] | None) -> float | None:
    """시세가 알려 주는 **마지막으로 마감된 정규장 종가**. 모르면 None.

    앵커 날짜(`bar_anchor`) 봉의 확정 종가가 필요한 자리에 쓴다. 가격 캐시의 그 봉을 그냥
    읽으면 안 된다 — 증분 배치가 장중에 그날 봉을 쓰고 실시간이 그 봉을 교체하므로,
    캐시의 마지막 봉은 **설계상 잠정값**이다. 미국은 정규장 마감(05:00 KST) 후 전체
    재정렬(17:10 KST)까지 12시간 동안 그 잠정 봉이 남는다 — 실측(2026-09-24 데이장)에서
    us_stock 20종목 전부 캐시 09-23 봉이 실제 종가와 달랐다(MRNA 184.27 vs 182.11).

    `prevClose` 는 일간 등락률의 비교 기준이며 세션에 따라 날짜가 달라진다.
    소스가 확정 종가를 주지 않는 경우 공통 가격 경로가 앵커 날짜의 캐시 봉으로 채운다.
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


def confirmed_close_on(close_series: pd.Series | None, bar_date: Any) -> float | None:
    """지정한 거래일의 확정 종가. 그 날짜의 봉이 없으면 값을 추정하지 않는다."""
    if close_series is None or close_series.empty:
        return None
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return None
    index = pd.DatetimeIndex(series.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    matched = series[index.normalize() == _normalize_day(bar_date)]
    if matched.empty:
        return None
    value = float(matched.iloc[-1])
    return value if value > 0 else None


def day_change_pct(close_series: pd.Series | None, bar_date: Any, *, anchor_close: float | None) -> float | None:
    """`bar_date` 봉의 전일 대비 변동률(%) — 전거래일 컬럼의 계산.

    **날짜로 찾는다.** 실시간을 반영한 시리즈의 뒤에서 두 번째 봉을 쓰면 장 시작 전
    (프리·데이장)에 하루 더 밀린다 — 그 구간에서는 실시간이 새 봉을 만들지 않고 마지막
    확정 봉을 **교체**하기 때문이다. 재현: MRNA 데이장에 09-23(-0.25%) 대신 09-22(+5.56%) 가 나왔다.

    분자는 시세가 주는 확정 종가(`anchor_close`)를 쓰고, 없을 때만 캐시의 그 날짜 봉으로
    내려간다 — 캐시의 마지막 봉은 설계상 잠정값이다(`last_regular_close`).
    분모는 그 날짜 **직전** 거래일의 캐시 봉이고, 이쪽은 이미 확정된 봉이다.
    """
    prev_close = confirmed_close_before(close_series, bar_date)
    if prev_close is None:
        return None

    if anchor_close is not None and anchor_close > 0:
        close = float(anchor_close)
    else:
        # 그 **날짜** 봉만 쓴다. 없는 종목(거래정지·상장 직후)은 비워 둔다 —
        # 며칠치 변동을 하루치로 표기하지 않는다.
        close = confirmed_close_on(close_series, bar_date)
        if close is None:
            return None
    return ((close / prev_close) - 1.0) * 100.0


def prior_day_shown(country_code: str, *, as_of_is_today: bool) -> bool:
    """전거래일 컬럼이 앵커 봉보다 **한 거래일 앞**을 보일지.

    일간(%)이 앵커 봉(마지막 마감 거래일)의 변동을 보이는 동안이다. 마감 세션이 그렇고, 애프터마켓 일간이
    그날 전체 변동인 시장(한국)은 애프터마켓도 그렇다. 네이버·토스처럼 다음 첫 세션 전까지 마지막 거래일
    변동을 유지한다. 과거 기준일이면 항상 거짓.
    """
    if not as_of_is_today:
        return False
    session = market_session(country_code)["session"]
    return session == CLOSED or (session == AFTERMARKET and aftermarket_in_daily_change(country_code))


def previous_day_change_pct(
    confirmed_close_series: pd.Series | None,
    realtime_entry: Mapping[str, Any] | None,
    country_code: str,
    *,
    prior_shown: bool,
) -> float | None:
    """전거래일(%) — 순위·모멘텀 화면 공용. `prior_shown` 은 `prior_day_shown` 의 결과.

    기본은 앵커 봉의 변동(일간이 이미 새 날을 보이는 구간)이고, 일간이 앵커 봉을 보이는
    동안(`prior_shown`)에는 그보다 한 거래일 앞 봉의 변동이다.
    """
    anchor = bar_anchor(country_code)
    if not prior_shown:
        return day_change_pct(confirmed_close_series, anchor, anchor_close=last_regular_close(realtime_entry))
    if confirmed_close_series is None or confirmed_close_series.empty:
        return None
    series = pd.to_numeric(confirmed_close_series, errors="coerce").dropna()
    prior = series.loc[pd.DatetimeIndex(series.index).normalize() < anchor]
    if prior.empty:
        return None
    return day_change_pct(series, pd.Timestamp(prior.index[-1]), anchor_close=None)


@dataclass(frozen=True)
class _PriceRule:
    """지금 세션에서 봉에 가격을 넣는 규칙 — 한 번만 계산해 종목마다 재사용한다."""

    kind: str  # regular(정규장 중) · pre_open(프리장·데이장, N+1 새 봉) · settled(애프터·마감, N 종가 확정)
    oldest_trade_day: pd.Timestamp  # 이보다 오래된 체결 시세는 버린다
    closed: bool  # 세션이 마감 상태 — 시세 `nowVal` 이 곧 정규장 종가다


def _price_rule(country_code: str, bar_day: pd.Timestamp, now: datetime | None) -> _PriceRule:
    session = market_session(country_code, now=now)["session"]
    if pre_open_trading_day(country_code, now=now) is not None:
        # 새 봉은 직전 마감 이후의 체결이면 받는다 — 시간외 체결이 없는 종목은 마지막 종가가 곧 현재가다.
        return _PriceRule("pre_open", bar_anchor(country_code, now=now), closed=False)
    if session == REGULAR:
        return _PriceRule("regular", bar_day, closed=False)
    return _PriceRule("settled", bar_day, closed=session == CLOSED)


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bar_price(entry: Mapping[str, Any] | None, rule: _PriceRule, *, display: bool) -> float | None:
    """붙여도 되는 가격. `display` 면 현재가 표시용(`nowVal`), 아니면 **봉에 넣는 판정용** 가격이다.

    판정용은 애프터·마감 구간에서 정규장 종가다 — 시간외 가격(`nowVal`)은 마감 봉을 바꾸지 못한다.
    마감 상태에서는 소스가 `nowVal` 을 정규장 종가로 확정해 주므로 그 값을 그대로 쓴다.

    **붙일 봉보다 오래된 시세는 버린다.** 시세 소스가 멈춰 있거나(그 세션에 체결이 없는
    종목) 캐시된 값을 주면 며칠 전 가격이 오늘 봉으로 들어간다 — 재현: 금요일 시세 99.0 이
    화요일 봉에 그대로 들어갔다. 거래 시각을 모르는 소스는 검증을 건너뛴다(위 함수 주석).
    """
    if not isinstance(entry, Mapping) or not entry:
        return None
    trade_day = quote_trade_day(entry)
    if trade_day is not None and trade_day < rule.oldest_trade_day:
        return None
    if display or rule.kind != "settled":
        return _positive(entry.get("nowVal"))
    settled = last_regular_close(entry) or _positive(entry.get("regularClose"))
    if settled is None and rule.closed:
        settled = _positive(entry.get("nowVal"))
    return settled


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
    reference = _normalize_day(last_bar)
    target = effective_bar_date(reference, country_code, now=now)
    price = _bar_price(realtime_entry, _price_rule(country_code, target, now), display=False)

    if cached_close_series is None or cached_close_series.empty:
        # 확정 봉이 하나도 없는 종목(상장 첫날 장중) — 실시간 봉 하나가 시리즈 전부다.
        if price is None:
            return None
        return pd.Series([price], index=pd.DatetimeIndex([target]))
    if price is None:
        return cached_close_series

    adjusted = cached_close_series.copy()
    adjusted.index = pd.DatetimeIndex([_normalize_day(idx) for idx in adjusted.index])

    adjusted.loc[target] = price
    return adjusted.sort_index()


def live_prices_for_bar(
    snapshot: Mapping[str, Mapping[str, Any]] | None,
    bar_date: Any,
    *,
    country: str,
    field: str = "nowVal",
    now: datetime | None = None,
) -> dict[str, float]:
    """스냅샷에서 **그 봉에 붙여도 되는** 값만 골라 `{티커: 값}` 으로 만든다.

    프레임 경로(`apply_realtime_closes`)는 이미 추출된 숫자를 받으므로, 신선도 검증은
    시세 항목이 남아 있는 이 자리에서 해야 한다 — 순위(단일 종목)와 전략(프레임)이
    같은 기준을 쓰도록 여기 한 곳에 둔다.

    `field` 는 종가 외 프레임(시가 등)을 만들 때 바꿔 쓴다. 프리장·데이장의 새 봉은 아직
    시가가 없으므로(스냅샷 시가는 직전 세션의 값) 시가는 비운다.
    """
    target = _normalize_day(bar_date)
    rule = _price_rule(country, target, now)
    if field != "nowVal" and rule.kind == "pre_open":
        return {}
    live: dict[str, float] = {}
    for ticker, entry in (snapshot or {}).items():
        if not isinstance(entry, Mapping):
            continue
        if field == "nowVal":
            value = _bar_price(entry, rule, display=False)
        else:
            trade_day = quote_trade_day(entry)
            if trade_day is not None and trade_day < target:
                continue
            value = _positive(entry.get(field))
        if value is not None:
            live[str(ticker)] = value
    return live


def live_display_prices(
    snapshot: Mapping[str, Mapping[str, Any]] | None,
    bar_date: Any,
    *,
    country: str,
    now: datetime | None = None,
) -> dict[str, float]:
    """현재가 **표시용** 가격 — 시간외 가격을 포함한 현재 세션 시세다(봉에 넣는 값과 구분한다)."""
    rule = _price_rule(country, _normalize_day(bar_date), now)
    prices: dict[str, float] = {}
    for ticker, entry in (snapshot or {}).items():
        value = _bar_price(entry, rule, display=True)
        if value is not None:
            prices[str(ticker)] = value
    return prices


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
