#!/usr/bin/env python
"""
Real Portfolio Performance Notifier.
Loads real holdings and cash from portfolio_master and sends performance alerts to Slack.
Supports individual account reporting and bulk processing for all registered accounts.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

import pytz

from utils.account_registry import get_account_settings, load_account_configs
from utils.data_loader import get_trading_days
from utils.env import load_env_if_present
from utils.notification import send_slack_message_v2
from utils.portfolio_io import load_portfolio_master, load_real_holdings_with_recommendations
from utils.report import format_kr_money

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress Streamlit warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
for _name in [
    "streamlit",
    "streamlit.runtime.caching.cache_data_api",
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.state.session_state_proxy",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)


def get_trend_emoji(val):
    if val > 0:
        return "📈"
    elif val < 0:
        return "📉"
    return ""


def get_profit_emoji(val):
    if val >= 3.0:
        return "🚀"
    elif val <= -3.0:
        return "😭"
    return ""


def process_account(account_id):
    """Processes portfolio data for a single account and returns a Slack message payload."""
    logger.info(f"Processing account: {account_id}")

    settings = get_account_settings(account_id)
    account_name = settings.get("name", account_id.upper())

    # 1. Load Real Holdings
    try:
        df = load_real_holdings_with_recommendations(account_id)
    except Exception as e:
        logger.error(f"Error loading holdings for {account_id}: {e}")
        return None

    # 2. Load Portfolio Master (Cash and Principal)
    m_data = load_portfolio_master(account_id)
    if not m_data:
        logger.warning(f"No master data found for {account_id}")
        return None

    principal = m_data.get("total_principal", 0.0)
    cash = m_data.get("cash_balance", 0.0)

    from utils.data_loader import count_trading_days

    now = datetime.now()
    valuation = 0.0
    holdings_text = []

    if df is not None and not df.empty:
        valuation = df["평가금액(KRW)"].sum()

        # Group by bucket
        buckets = df.groupby("버킷")
        for bucket_name, group in sorted(buckets):
            holdings_text.append(f"*{bucket_name}*")
            for _, row in group.iterrows():
                ticker = row["티커"]
                name = row["종목명"]
                profit_pct = row["수익률(%)"]
                profit_krw = row["평가손익(KRW)"]
                import pandas as pd

                daily_pct = row.get("일간(%)", 0.0)
                if pd.isna(daily_pct):
                    daily_pct = 0.0

                eval_amount = row["평가금액(KRW)"]

                # Calculate daily KRW change
                daily_krw = 0
                if abs(daily_pct) > 1e-6:
                    daily_krw = int(round(eval_amount * (daily_pct / 100) / (1 + daily_pct / 100)))

                currency = row.get("환종", "KRW")
                market = {"USD": "us", "AUD": "au"}.get(currency, "kor")

                # Calculate trading days
                try:
                    first_buy = row.get("first_buy_date")
                    if first_buy:
                        trading_days_held = count_trading_days(market, first_buy, now)
                    else:
                        trading_days_held = 1
                except Exception:
                    trading_days_held = row.get("보유일", 0)

                emoji_daily = get_trend_emoji(daily_pct)
                emoji_total = get_trend_emoji(profit_pct)
                prof_emoji_daily = get_profit_emoji(daily_pct)

                line = (
                    f"• {ticker} - {name} (보유일: {trading_days_held}거래일)\n"
                    f"누적 - {emoji_total} {profit_pct:+.2f}% ({format_kr_money(profit_krw)}) | "
                    f"일간 - {emoji_daily} {daily_pct:+.2f}% ({format_kr_money(daily_krw)}) {prof_emoji_daily}"
                )
                holdings_text.append(line)
            holdings_text.append("")

    total_assets = valuation + cash
    net_profit = total_assets - principal
    net_profit_pct = (net_profit / principal * 100) if principal > 0 else 0.0

    # 3. Daily Change Calculation
    from utils.portfolio_io import get_latest_daily_snapshot

    prev_snapshot = get_latest_daily_snapshot(account_id, before_today=True)
    daily_delta = 0.0
    daily_delta_pct = 0.0
    has_prev = False
    if prev_snapshot:
        prev_assets = prev_snapshot.get("total_assets", 0.0)
        if prev_assets > 0:
            daily_delta = total_assets - prev_assets
            daily_delta_pct = (daily_delta / prev_assets) * 100
            has_prev = True

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    daily_delta_val_text = "N/A"
    if has_prev:
        delta_emoji = get_trend_emoji(daily_delta)
        daily_delta_val_text = f"{delta_emoji} *{daily_delta_pct:+.2f}%* ({format_kr_money(daily_delta)})"

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"[Portfolio] {account_name}", "emoji": True}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*기준 시간*: {now_str}"},
                {"type": "mrkdwn", "text": f"*원금*: {format_kr_money(principal)}"},
                {"type": "mrkdwn", "text": f"*총 자산*: {format_kr_money(total_assets)}"},
                {
                    "type": "mrkdwn",
                    "text": f"*합계 수익*: {get_trend_emoji(net_profit)} *{net_profit_pct:+.2f}%* ({format_kr_money(net_profit)})",
                },
                {"type": "mrkdwn", "text": f"*전일 대비*: {daily_delta_val_text}"},
                {"type": "mrkdwn", "text": f"*현금*: {format_kr_money(cash)}"},
            ],
        },
        {"type": "divider"},
    ]

    if holdings_text:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(holdings_text)}})

    return {"text": f"[{account_name}] 실자산 현황 업데이트", "blocks": blocks}


def get_today_str_for_country(country: str) -> str:
    from config import MARKET_SCHEDULES

    schedule = MARKET_SCHEDULES.get(country.lower())
    if schedule and schedule.get("timezone"):
        tz = pytz.timezone(schedule["timezone"])
        return datetime.now(tz).strftime("%Y-%m-%d")
    return datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d")


def main():
    load_env_if_present()

    parser = argparse.ArgumentParser(description="Real Portfolio Status Notifier")
    parser.add_argument("account", nargs="?", help="Account ID (optional, runs all if omitted)")
    parser.add_argument("--country", help="Country code to filter by (e.g. kor, us, au)")
    args = parser.parse_args()

    target_country = args.country.lower() if args.country else None

    if args.account:
        # 1. 특정 계정 실행 시 해당 계정의 country_code 확인 후 휴장일 체크
        settings = get_account_settings(args.account)
        country = settings.get("country_code", "kor").strip().lower()

        today_str = get_today_str_for_country(country)
        try:
            trading_days = get_trading_days(today_str, today_str, country)
            if len(trading_days) == 0:
                logger.info(f"[{country.upper()}] 오늘은 휴장일입니다. ({today_str}) 작업을 건너뜁니다.")
                return
        except Exception as e:
            logger.warning(f"휴장일 조회 실패: {e}. 안전을 위해 영업일로 간주하고 진행합니다.")

        # Run single account
        payload = process_account(args.account)
        if payload:
            # Print to console
            print(f"\n--- Console Output: {args.account} ---")
            print(payload["text"])
            for block in payload.get("blocks", []):
                if block.get("text"):
                    print(block["text"].get("text", ""))
                if block.get("fields"):
                    for f in block["fields"]:
                        print(f.get("text", ""))

            # Send to Slack
            send_slack_message_v2(payload["text"], blocks=payload["blocks"])
            logger.info(f"Sent Slack message for {args.account}")
    else:
        # 2. 벌크 실행 시 국가 필터가 있으면 해당 국가 휴장일 체크
        if target_country:
            today_str = get_today_str_for_country(target_country)
            try:
                trading_days = get_trading_days(today_str, today_str, target_country)
                if len(trading_days) == 0:
                    logger.info(f"[{target_country.upper()}] 오늘은 휴장일입니다. ({today_str}) 작업을 건너뜁니다.")
                    return
            except Exception as e:
                logger.warning(f"휴장일 조회 실패: {e}. 안전을 위해 영업일로 간주하고 진행합니다.")

        # Run filtered or all accounts
        accounts = load_account_configs()
        for acc in accounts:
            account_id = acc["account_id"]
            country_code = acc.get("country_code", "").lower()

            # Filter by country if specified
            if target_country and country_code != target_country:
                continue

            if not acc.get("settings", {}).get("show_hold", True):
                continue

            payload = process_account(account_id)
            if payload:
                send_slack_message_v2(payload["text"], blocks=payload["blocks"])
                logger.info(f"Sent Slack message for {account_id}")


if __name__ == "__main__":
    main()
