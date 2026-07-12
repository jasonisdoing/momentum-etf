#!/usr/bin/env python
"""24H 실시간 주식 및 선물 시세를 슬랙으로 전송.

한국 종목은 KRW(환율 환산), 미국 종목은 USD 로 현재가/24h 변동률/실제가 대비 차이를 보낸다.
하이퍼리퀴드 선물 시세를 표기한다.
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LIVE_24H_ALERT_PCT
from utils.env import load_env_if_present
from utils.live_24h_service import load_live_24h_quotes
from utils.notification import send_slack_message_v2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _fmt_price(value, currency):
    if value is None:
        return "-"
    if currency == "KRW":
        return f"{round(value):,}원"
    if currency == "POINT":
        return f"{value:,.2f}p"
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


def _select_representative(quotes_by_symbol, toss_symbol, hyperliquid_symbol):
    """화면과 같은 신선도 기준으로 토스 또는 Hyperliquid 대표값을 선택한다."""
    toss_quote = quotes_by_symbol.get(toss_symbol)
    hyperliquid_quote = quotes_by_symbol.get(hyperliquid_symbol)
    if toss_quote and _has_fresh_toss_candle(toss_quote):
        return toss_quote, "토스", toss_quote.get("diff_pct")
    if not hyperliquid_quote:
        raise RuntimeError(f"대표 시세가 없습니다: {hyperliquid_symbol}")
    return hyperliquid_quote, "하이퍼리퀴드", hyperliquid_quote.get("change_24h_pct")


def main():
    load_env_if_present()
    data = load_live_24h_quotes()
    quotes = data.get("quotes", [])
    quotes_by_symbol = {str(quote.get("symbol") or ""): quote for quote in quotes}

    rows = []
    for symbol, name in (("NQ_FUT", "나스닥 100 선물"), ("USDKRW", "달러 환율"), ("VIX", "VIX")):
        quote = quotes_by_symbol.get(symbol)
        if not quote:
            raise RuntimeError(f"필수 시장지표 시세가 없습니다: {symbol}")
        rows.append((":us:", name, symbol, quote, "실시간", quote.get("diff_pct")))

    for flag, name, toss_symbol, hyperliquid_symbol in (
        (":kr:", "SK하이닉스", "SKHX_KR_TOSS", "SKHX"),
        (":kr:", "삼성전자", "SMSN_KR_TOSS", "SMSN"),
        (":us:", "마이크론", "MU_TOSS", "MU"),
    ):
        quote, source, change_pct = _select_representative(quotes_by_symbol, toss_symbol, hyperliquid_symbol)
        rows.append((flag, name, source, quote, None, change_pct))

    alerts = []  # 최근 1시간 |변동| ≥ 임계 인 종목 (name, move)
    body = []
    for flag, name, identifier, quote, status, change_pct in rows:
        m1 = _recent_move(quote.get("candles"), 1)
        triggered = m1 is not None and abs(m1) >= LIVE_24H_ALERT_PCT
        if triggered:
            alerts.append((name, m1))

        body.append(
            f"{flag} *{name}*({identifier}) *{_fmt_pct(change_pct)}*"
            f"{' (' + status + ')' if status else ''} {_trend_emoji(change_pct)}"
            f"{' 🚨' if triggered else ''}"
        )

    lines = []
    # 최근 1시간 급변 종목이 있으면 맨 위에 @channel 핑
    if alerts:
        tags = ", ".join(f"{name} {mv:+.1f}%" for name, mv in alerts)
        lines.append(f"<!channel> 🚨 *최근 1시간 급변* — {tags}")
    lines.extend(body)
    # 타이틀(링크)은 목록 아래에 — 클릭 시 live-24h 페이지로 이동
    lines.append("*<https://etf.dojason.com/live-24h|🌐 24H 시세>*")

    send_slack_message_v2("\n".join(lines))
    logger.info("24H 시세 슬랙 전송 완료 (%d종목, 1시간 급변 %d건)", len(rows), len(alerts))


if __name__ == "__main__":
    main()
