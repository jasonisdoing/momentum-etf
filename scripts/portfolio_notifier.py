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

from utils.account_registry import get_account_settings, load_account_configs
from utils.env import load_env_if_present
from utils.notification import send_slack_message_v2
from utils.portfolio_io import load_portfolio_master, load_real_holdings_with_recommendations
from utils.report import format_kr_money

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_trend_emoji(val):
    if val > 0:
        return "📈"
    elif val < 0:
        return "📉"
    return ""


def get_profit_emoji(val):
    if val > 0:
        return "🚀"
    elif val < -3.0:
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

                emoji = get_trend_emoji(profit_pct)
                prof_emoji = get_profit_emoji(profit_pct)

                line = (
                    f"• *{ticker}* - {name} {emoji} *{profit_pct:+.2f}%* ({format_kr_money(profit_krw)}) {prof_emoji}"
                )
                holdings_text.append(line)
            holdings_text.append("")

    total_assets = valuation + cash
    net_profit = total_assets - principal
    net_profit_pct = (net_profit / principal * 100) if principal > 0 else 0.0

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"实 [Portfolio] {account_name}", "emoji": True}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*기준일*: {now_str}"},
                {"type": "mrkdwn", "text": f"*총 자산*: {format_kr_money(total_assets)}"},
            ],
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*원금*: {format_kr_money(principal)}"},
                {"type": "mrkdwn", "text": f"*현금*: {format_kr_money(cash)}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*합계 수익*: {get_trend_emoji(net_profit)} *{net_profit_pct:+.2f}%* ({format_kr_money(net_profit)})",
            },
        },
        {"type": "divider"},
    ]

    if holdings_text:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(holdings_text)}})

    return {"text": f"[{account_name}] 실자산 현황 업데이트", "blocks": blocks}


def main():
    load_env_if_present()

    parser = argparse.ArgumentParser(description="Real Portfolio Status Notifier")
    parser.add_argument("account", nargs="?", help="Account ID (optional, runs all if omitted)")
    parser.add_argument("--country", help="Country code to filter by (e.g. kor, us, au)")
    args = parser.parse_args()

    if args.account:
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
        # Run filtered or all accounts
        accounts = load_account_configs()
        target_country = args.country.lower() if args.country else None

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
