#!/usr/bin/env python
"""24H 실시간 주식 및 선물 시세를 슬랙으로 전송.

한국 종목은 KRW(환율 환산), 미국 종목은 USD 로 현재가/24h 변동률/실제가 대비 차이를 보낸다.
하이퍼리퀴드 선물 시세를 표기한다.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main():
    load_env_if_present()
    data = load_live_24h_quotes()
    quotes = data.get("quotes", [])

    lines = []

    # 1. 하이퍼리퀴드 시세 블록
    lines.append("*🌐 하이퍼리퀴드 24H 시세*")
    for q in quotes:
        flag = ":kr:" if q.get("country") == "kor" else ":us:"
        currency = q.get("currency", "USD")
        hl_price = q.get("hyper_price")
        hl_change = q.get("change_24h_pct")
        hl_diff = q.get("diff_pct")  # 정규장 대비 (메인)

        lines.append(
            f"{flag} *{q['name']}*({q['symbol']}) {_fmt_pct(hl_diff)}({_fmt_price(hl_price, currency)}) {_trend_emoji(hl_diff)}\n"
            f"   • 24시간: {_fmt_pct(hl_change)} {_trend_emoji(hl_change)}"
        )



    send_slack_message_v2("\n".join(lines))
    logger.info("24H 시세 슬랙 전송 완료 (%d종목)", len(quotes))


if __name__ == "__main__":
    main()
