#!/usr/bin/env python
"""Hyperliquid 24시간 토큰화 주식 시세(삼성전자/SK하이닉스/마이크론)를 슬랙으로 전송.

한국 종목은 KRW(환율 환산), 미국 종목은 USD 로 현재가/24h 변동률/실제가 대비 차이를 보낸다.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import load_env_if_present
from utils.hyperliquid_service import load_hyperliquid_quotes
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


def main():
    load_env_if_present()
    data = load_hyperliquid_quotes()

    lines = ["*🌐 하이퍼리퀴드 24시간 시세*"]
    for q in data.get("quotes", []):
        flag = "🇰🇷" if q.get("country") == "kor" else "🇺🇸"
        currency = q.get("currency", "USD")
        lines.append(
            f"{flag} *{q['name']}*: "
            f"{_fmt_price(q.get('hyper_price'), currency)} "
            f"(*{_fmt_pct(q.get('change_24h_pct'))}*) {_trend_emoji(q.get('change_24h_pct'))}\n"
            f"   • 실제가 대비: {_fmt_pct(q.get('diff_pct'))}"
        )

    send_slack_message_v2("\n".join(lines))
    logger.info("Hyperliquid 슬랙 전송 완료 (%d종목)", len(data.get("quotes", [])))


if __name__ == "__main__":
    main()
