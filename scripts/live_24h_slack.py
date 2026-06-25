#!/usr/bin/env python
"""24H 실시간 주식 및 선물 시세를 슬랙으로 전송.

한국 종목은 KRW(환율 환산), 미국 종목은 USD 로 현재가/24h 변동률/실제가 대비 차이를 보낸다.
하이퍼리퀴드 선물 시세를 표기한다.
"""

import logging
import os
import sys

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
    """30분봉 기준 최근 N시간 변동률(%). 2N봉 전 대비. 데이터 부족 시 None."""
    if not candles:
        return None
    idx = 2 * hours  # N시간 = 2N개의 30분봉
    if len(candles) <= idx:
        return None
    prev = candles[-1 - idx].get("c")
    cur = candles[-1].get("c")
    if not prev or not cur:
        return None
    return (cur / prev - 1.0) * 100.0


def main():
    load_env_if_present()
    data = load_live_24h_quotes()
    quotes = data.get("quotes", [])

    alerts = []  # 최근 1시간 |변동| ≥ 임계 인 종목 (name, move)
    body = []
    for q in quotes:
        flag = ":kr:" if q.get("country") == "kor" else ":us:"
        hl_diff = q.get("diff_pct")  # 정규장 종가 대비 (메인)

        m1 = _recent_move(q.get("candles"), 1)
        triggered = m1 is not None and abs(m1) >= LIVE_24H_ALERT_PCT
        if triggered:
            alerts.append((q["name"], m1))

        session = "장중" if q.get("session_open") else "시간외"
        body.append(
            f"{flag} *{q['name']}*({q['symbol']}) *{_fmt_pct(hl_diff)}* ({session}) {_trend_emoji(hl_diff)}"
            f"{' 🚨' if triggered else ''}"
        )

    lines = []
    # 최근 1시간 급변 종목이 있으면 맨 위에 @channel 핑
    if alerts:
        tags = ", ".join(f"{name} {mv:+.1f}%" for name, mv in alerts)
        lines.append(f"<!channel> 🚨 *최근 1시간 급변* — {tags}")
    # 헤더 클릭 시 live-24h 페이지로 이동
    lines.append("*<https://etf.dojason.com/live-24h|🌐 하이퍼리퀴드 24H 시세>*")
    lines.extend(body)

    send_slack_message_v2("\n".join(lines))
    logger.info("24H 시세 슬랙 전송 완료 (%d종목, 1시간 급변 %d건)", len(quotes), len(alerts))


if __name__ == "__main__":
    main()
