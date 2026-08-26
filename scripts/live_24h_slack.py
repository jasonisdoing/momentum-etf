#!/usr/bin/env python
"""24H 실시간 주식 및 선물 시세를 슬랙으로 전송.

종목별 거래 세션에 따라 토스 또는 Hyperliquid 대표 시세를 선택해 현재가와 변동률을 보낸다.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LIVE_24H_ALERT_PCT
from utils.env import load_env_if_present
from utils.live_24h_service import load_live_24h_quotes
from utils.notification import app_link, send_slack_message_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fmt_price(value, currency):
    if value is None:
        return "-"
    if currency == "KRW":
        return f"{round(value):,}원"
    if currency == "POINT":
        return f"{value:,.2f}p"
    if currency == "FX":
        return f"{value:,.2f}원"
    return f"${value:,.2f}"


def _fmt_pct(value):
    return "-" if value is None else f"{value:+.2f}%"


def _trend_emoji(value):
    if value is None or value == 0:
        return ""
    return ":small_red_triangle:" if value > 0 else ":chart_with_downwards_trend:"


def _recent_move(candles, hours):
    """15분봉 기준 최근 N시간 변동률(%). 4N봉 전 대비. 데이터 부족 시 None."""
    if not candles:
        return None
    idx = 4 * hours
    if len(candles) <= idx:
        return None
    prev = candles[-1 - idx].get("c")
    cur = candles[-1].get("c")
    if not prev or not cur:
        return None
    return (cur / prev - 1.0) * 100.0


def _has_fresh_toss_candle(quote):
    """토스의 마지막 15분봉이 현재 기준 30분 이내인지 확인한다."""
    candles = quote.get("candles") or []
    if not candles:
        return False
    timestamp = candles[-1].get("t")
    return timestamp is not None and timestamp >= time.time() * 1000 - 30 * 60 * 1000


def _can_use_toss_as_representative(quote):
    """거래 세션이 열려 있고 최근 토스 캔들이 있을 때만 대표값으로 사용한다."""
    return bool(
        quote
        and quote.get("price_data_open")
        and quote.get("price_data_session") != "closed"
        and _has_fresh_toss_candle(quote)
    )


def _hyperliquid_change_from_previous_close(quote):
    """기초 종목의 직전 거래일 종가 대비 Hyperliquid 변동률을 계산한다."""
    price = quote.get("hyper_price")
    previous_close = quote.get("reference_prev_close")
    if price is None or not previous_close or previous_close <= 0:
        return None
    return (price / previous_close - 1.0) * 100.0


def _toss_source_label(quote):
    """토스 대표 시세의 국가와 거래 세션을 슬랙 표시명으로 반환한다."""
    country = "한국" if quote.get("country") == "kor" else "미국"
    session = {
        "daymarket": "데이",
        "premarket": "프리",
        "regular": "본장",
        "aftermarket": "애프터",
    }.get(quote.get("price_data_session"))
    if not session:
        raise RuntimeError(f"토스 대표 시세의 거래 세션이 올바르지 않습니다: {quote.get('price_data_session')}")
    return f"{country} {session}"


def _select_representative(quotes_by_symbol, toss_symbol, hyperliquid_symbol):
    """화면과 같은 거래 세션·신선도 기준으로 대표값을 선택한다."""
    toss_quote = quotes_by_symbol.get(toss_symbol)
    hyperliquid_quote = quotes_by_symbol.get(hyperliquid_symbol)
    if _can_use_toss_as_representative(toss_quote):
        return toss_quote, _toss_source_label(toss_quote), toss_quote.get("diff_pct")
    if not hyperliquid_quote:
        raise RuntimeError(f"대표 시세가 없습니다: {hyperliquid_symbol}")
    return hyperliquid_quote, "하이퍼리퀴드", _hyperliquid_change_from_previous_close(hyperliquid_quote)


def main():
    load_env_if_present()
    data = load_live_24h_quotes()
    quotes = data.get("quotes", [])
    quotes_by_symbol = {str(quote.get("symbol") or ""): quote for quote in quotes}

    rows = []
    # VIX 는 슬랙 알림 대상에서 제외한다(/live-24h 화면에는 그대로 표시).
    for symbol, name in (("NQ_FUT", "나스닥 100 선물"), ("USDKRW", "달러 환율")):
        quote = quotes_by_symbol.get(symbol)
        if not quote:
            raise RuntimeError(f"필수 시장지표 시세가 없습니다: {symbol}")
        rows.append((":us:", name, symbol, quote, "실시간", quote.get("diff_pct"), symbol != "NQ_FUT"))

    for flag, name, toss_symbol, hyperliquid_symbol in (
        (":kr:", "SK하이닉스", "SKHX_KR_TOSS", "SKHX"),
        (":kr:", "삼성전자", "SMSN_KR_TOSS", "SMSN"),
        (":us:", "마이크론", "MU_TOSS", "MU"),
    ):
        quote, source, change_pct = _select_representative(quotes_by_symbol, toss_symbol, hyperliquid_symbol)
        rows.append((flag, name, source, quote, None, change_pct, True))

    alerts = []  # 최근 1시간 |변동| ≥ 임계 인 종목 (name, move)
    body = []
    for flag, name, identifier, quote, status, change_pct, show_price in rows:
        m1 = _recent_move(quote.get("candles"), 1)
        triggered = m1 is not None and abs(m1) >= LIVE_24H_ALERT_PCT
        if triggered:
            alerts.append((name, m1))

        price_text = f" *{_fmt_price(quote.get('hyper_price'), quote.get('currency'))}*" if show_price else ""
        body.append(
            f"{flag} *{name}*({identifier}){price_text} *{_fmt_pct(change_pct)}*"
            f"{' (' + status + ')' if status else ''} {_trend_emoji(change_pct)}"
            f"{' 🚨' if triggered else ''}"
        )

    # 급변 종목이 없으면 보내지 않는다 — 알릴 것이 있을 때만(@channel) 발송한다.
    if not alerts:
        logger.info("24H 시세 급변 없음 — 슬랙 발송 생략 (%d종목, 임계 %.1f%%)", len(rows), LIVE_24H_ALERT_PCT)
        return

    # 최근 1시간 급변 종목을 맨 위에 @channel 핑으로 알린다.
    tags = ", ".join(f"{name} {mv:+.1f}%" for name, mv in alerts)
    lines = [f"<!channel> 🚨 *최근 1시간 급변* — {tags}"]
    lines.extend(body)

    send_slack_message_v2("\n".join(lines))
    logger.info("24H 시세 슬랙 전송 완료 (%d종목, 1시간 급변 %d건)", len(rows), len(alerts))


if __name__ == "__main__":
    main()
