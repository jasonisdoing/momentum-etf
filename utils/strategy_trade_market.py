"""사고팔기의 시장별 규칙 — 금액 표기·주문일자·주문 거래소·마감 경고를 국가(kor·us)별로 한 곳에서 정한다.

주문·원장·알림이 "kor" 를 직접 적지 않고 이 모듈을 거친다. 국장은 KRX·NXT 두 거래소가 거래 시간으로 갈리고,
미장은 거래소(나스닥·뉴욕·아멕스)가 종목마다 정해져 있으며 프리·정규·애프터·데이장 세션이 모두 같은 지정가 주문으로 낸다.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from config import MARKET_SCHEDULES
from utils.market_session import (
    _in_us_daymarket,
    is_instrument_trading_time,
    kor_order_venue,
)

#: 국장 거래소별 마감 경고 구간 — 이 시각부터 그 거래소 마감까지 아직 체결되지 않은 슬랙 주문을 한 번 경고한다.
#: KRX 는 정규장 마감 15:30, NXT 는 애프터마켓 마감 20:00(개별주의 정규장 밖 주문).
KOR_CLOSE_WARNING_WINDOWS = {"KRX": (time(15, 20), time(15, 30)), "NXT": (time(19, 50), time(20, 0))}
#: 국장 거래소별로 주문이 살아 있는 거래 시간표의 종목 종류 — KRX 는 정규장(ETF 시간표), NXT 는 개별주 시간표.
KOR_VENUE_WINDOW_KIND = {"KRX": "etf", "NXT": "stock"}
#: 미장 마감 경고 구간(ET) — 애프터마켓 마감 20:00 전, 데이장 마감 03:50 전 각 10분.
US_CLOSE_WARNING_WINDOWS = ((time(19, 50), time(20, 0)), (time(3, 40), time(3, 50)))


def _zone(country: str) -> ZoneInfo:
    return ZoneInfo(str(MARKET_SCHEDULES[country]["timezone"]))


def format_level(value: float | None, country: str) -> str:
    """단위 없는 가격 — 국장 원 단위 정수, 미장 달러 소수 둘째 자리. 값이 없으면 '-'."""
    if value is None:
        return "-"
    return f"{value:,.0f}" if country == "kor" else f"{value:,.2f}"


def format_money(value: float | None, country: str) -> str:
    """단위를 붙인 가격 — `1,234원` · `$12.34`. 값이 없으면 '-'."""
    if value is None:
        return "-"
    return f"{value:,.0f}원" if country == "kor" else f"${value:,.2f}"


def krw_rate(country: str) -> float:
    """종목 통화→원 환율 — 국장 1.0, 미장은 달러 환율. 못 받으면 예외(임의 환율을 쓰지 않는다)."""
    if country == "kor":
        return 1.0
    from services.price_service import get_exchange_rates

    rate = float(((get_exchange_rates(["USD"]) or {}).get("USD") or {}).get("rate") or 0.0)
    if rate <= 0:
        raise ValueError("달러 환율을 받을 수 없어 회차 수량을 계산할 수 없습니다.")
    return rate


def broker_order_date(country: str, created: datetime) -> str:
    """증권사 주문내역이 쓰는 주문일자(YYYYMMDD) — 그 시장의 현지 달력 날짜다(미국 애프터 주문은 한국 날짜로 전날)."""
    return created.astimezone(_zone(country)).strftime("%Y%m%d")


def session_trade_date(country: str, at: datetime) -> date:
    """주문이 속한 **거래 세션의 날짜** — 만료 판정용. 미국 데이장은 20:00 ET 이후라도 다음 거래일 세션에 속한다."""
    local = at.astimezone(_zone(country))
    if country == "us" and (daymarket_day := _in_us_daymarket(local)) is not None:
        return daymarket_day
    return local.date()


def order_venue(country: str, kind: str, exchange: str | None, *, now: datetime) -> str:
    """지금 이 종목에 주문을 낼 거래소 — 거래 시간 밖이면 ValueError.

    국장은 정규장 KRX·개별주 정규장 밖 NXT, 미장은 종목의 거래소 코드(`exchange`)다.
    """
    if country == "kor":
        return kor_order_venue(kind, now=now)
    if not exchange:
        raise ValueError("이 종목의 거래소가 저장돼 있지 않습니다 — 전략을 다시 저장하세요.")
    if not is_instrument_trading_time(country, kind, now=now):
        raise ValueError("지금은 이 종목의 거래 시간이 아닙니다.")
    return exchange


def is_order_open(country: str, venue: str, kind: str, *, now: datetime) -> bool:
    """그 주문이 낸 거래소가 지금도 거래 중인지 — 끝났으면 미체결 당일 주문은 만료다."""
    if country == "kor":
        return is_instrument_trading_time("kor", KOR_VENUE_WINDOW_KIND[venue], now=now)
    return is_instrument_trading_time(country, kind, now=now)


def closing_windows(country: str, kind: str, venue: str, *, now: datetime) -> tuple[time, time] | None:
    """지금이 그 거래소의 마감 경고 구간이면 (시작, 종료) 현지 시각, 아니면 None."""
    local = now.astimezone(_zone(country))
    if country == "kor":
        start, end = KOR_CLOSE_WARNING_WINDOWS[venue]
        windows = ((start, end),)
        window_kind = KOR_VENUE_WINDOW_KIND[venue]
    else:
        windows = US_CLOSE_WARNING_WINDOWS
        window_kind = kind
    if not is_instrument_trading_time(country, window_kind, now=now):
        return None
    return next(((start, end) for start, end in windows if start <= local.time() < end), None)


def close_text(country: str, venue: str, *, now: datetime) -> str:
    """마감 경고 문구용 — `15:30 KRX` · `20:00 ET`."""
    if country == "kor":
        return f"{KOR_CLOSE_WARNING_WINDOWS[venue][1]:%H:%M} {venue}"
    local = now.astimezone(_zone(country)).time()
    end = next(end for start, end in US_CLOSE_WARNING_WINDOWS if start <= local < end)
    return f"{end:%H:%M} ET"
