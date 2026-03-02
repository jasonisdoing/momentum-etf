#!/usr/bin/env python
"""
Total Asset Summary Slack Notifier.
Aggregates data from all accounts and sends a summary message to Slack,
with detailed accounts and portfolio composition in a threaded reply.
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.account_registry import load_account_configs
from utils.env import load_env_if_present
from utils.notification import send_slack_message_v2
from utils.portfolio_io import (
    get_latest_daily_snapshot,
    load_portfolio_master,
    load_real_holdings_with_recommendations,
    save_daily_snapshot,
)

# Suppress Streamlit warnings in non-Streamlit environments
os.environ["STREAMLIT_GLOBAL_LOG_LEVEL"] = "error"
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

# Silencing the "missing ScriptRunContext" and other Streamlit UserWarnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# streamlit.runtime.scriptrunner_utils.script_run_context
try:
    from streamlit.runtime.scriptrunner_utils import script_run_context

    script_run_context._LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def format_korean_currency(value):
    """원화 값을 '억/만원' 단위로 변환합니다."""
    if value == 0:
        return "0원"

    abs_value = abs(value)
    eok = int(abs_value // 100_000_000)
    remain = abs_value % 100_000_000
    man = int(round(remain / 10_000))

    if man == 10000:
        eok += 1
        man = 0

    parts = []
    if eok > 0:
        parts.append(f"{eok}억")
    if man > 0:
        parts.append(f"{man}만원")

    result = " ".join(parts) if parts else "0원"
    if value < 0:
        result = "-" + result
    return result


def get_trend_emoji(val):
    if val > 0:
        return "🔺"
    elif val < 0:
        return "🔹"
    return ""


def get_chart_emoji(val):
    if val > 0:
        return "📈"
    elif val < 0:
        return "📉"
    return "📊"


def main():
    load_env_if_present()

    accounts = load_account_configs()
    if not accounts:
        logger.error("No account configurations found.")
        return

    all_holdings = []
    account_summaries = []
    global_principal = 0.0
    global_cash = 0.0

    logger.info("Aggregating data from %d accounts...", len(accounts))

    for account in accounts:
        account_id = account["account_id"]
        if not account.get("settings", {}).get("show_hold", True):
            continue

        account_name = account.get("name") or account_id.upper()

        # Load principal and cash
        m_data = load_portfolio_master(account_id)
        if m_data:
            global_principal += m_data.get("total_principal", 0.0)
            global_cash += m_data.get("cash_balance", 0.0)

        # Load holdings
        try:
            # We need to mock streamlit session_state/secrets for some utils if they depend on it
            # But load_real_holdings_with_recommendations might work if handled carefully
            df = load_real_holdings_with_recommendations(account_id)
        except Exception as e:
            error_msg = f"❌ 자산 요약 생성 중 치명적 에러 발생 ({account_id} 계좌):\n```{e}```\n\n잘못된 자산 리포트가 발송되는 것을 방지하기 위해 오늘 알림을 중단합니다."
            logger.error(error_msg)
            send_slack_message_v2(error_msg)
            sys.exit(1)

        acc_valuation = 0.0
        acc_purchase = 0.0
        if df is not None and not df.empty:
            all_holdings.append(df)
            acc_valuation = df["평가금액(KRW)"].sum()
            acc_purchase = df["매입금액(KRW)"].sum()

        acc_principal = m_data.get("total_principal", 0.0) if m_data else 0.0
        acc_cash = m_data.get("cash_balance", 0.0) if m_data else 0.0
        acc_total_assets = acc_valuation + acc_cash
        acc_net_profit = acc_total_assets - acc_principal
        acc_net_profit_pct = (acc_net_profit / acc_principal * 100) if acc_principal > 0 else 0.0

        acc_stock_profit = acc_valuation - acc_purchase
        acc_stock_profit_pct = (acc_stock_profit / acc_purchase * 100) if acc_purchase > 0 else 0.0

        if acc_principal > 0 or acc_cash > 0 or acc_valuation > 0:
            account_summaries.append(
                {
                    "account_id": account_id,
                    "name": account_name,
                    "principal": acc_principal,
                    "total_assets": acc_total_assets,
                    "net_profit": acc_net_profit,
                    "net_profit_pct": acc_net_profit_pct,
                    "valuation": acc_valuation,
                    "stock_profit": acc_stock_profit,
                    "stock_profit_pct": acc_stock_profit_pct,
                    "cash": acc_cash,
                }
            )

    if not account_summaries:
        logger.warning("No data found to report.")
        return

    # Global Calculations
    total_assets = sum(acc["total_assets"] for acc in account_summaries)
    total_net_profit = total_assets - global_principal
    total_net_profit_pct = (total_net_profit / global_principal * 100) if global_principal > 0 else 0.0

    # Fetch previous snapshots
    prev_global = get_latest_daily_snapshot("TOTAL", before_today=True)
    global_change = 0.0
    global_change_pct = 0.0
    if prev_global:
        prev_total = prev_global.get("total_assets", 0.0)
        if prev_total > 0:
            global_change = total_assets - prev_total
            global_change_pct = (global_change / prev_total) * 100

    # 1. Compose Main Message (Total Summary)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    main_text = (
        f"*📊 총 자산 요약 ({now_str})*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *총 자산*: *{format_korean_currency(total_assets)}*\n"
        f"🏛️ *투자 원금*: {format_korean_currency(global_principal)}\n"
        f"💵 *현금 잔고*: {format_korean_currency(global_cash)}\n"
        f"{get_chart_emoji(global_change)} *전일 대비*: {format_korean_currency(global_change)} ({global_change_pct:+.2f}%) {get_trend_emoji(global_change)}\n"
        f"{get_chart_emoji(total_net_profit)} *총 평가손익*: *{format_korean_currency(total_net_profit)} ({total_net_profit_pct:+.2f}%)*\n"
    )

    main_ts = send_slack_message_v2(main_text)
    if not main_ts:
        logger.error("Failed to send main Slack message.")
        return

    # 2. Compose Account Details (Thread)
    acc_details = ["*📂 계좌별 상세 현황*"]
    for acc in account_summaries:
        # Fetch previous account snapshot
        prev_acc = get_latest_daily_snapshot(acc["account_id"], before_today=True)
        acc_change = 0.0
        acc_change_pct = 0.0
        if prev_acc:
            prev_acc_total = prev_acc.get("total_assets", 0.0)
            if prev_acc_total > 0:
                acc_change = acc["total_assets"] - prev_acc_total
                acc_change_pct = (acc_change / prev_acc_total) * 100

        emoji = get_trend_emoji(acc["net_profit"])
        change_emoji = get_trend_emoji(acc_change)
        line = (
            f"• *{acc['name']}*\n"
            f"  - 자산: {format_korean_currency(acc['total_assets'])} (원금: {format_korean_currency(acc['principal'])})\n"
            f"  - 수익: {emoji} {acc['net_profit_pct']:+.2f}% ({format_korean_currency(acc['net_profit'])})\n"
            f"  - 변동: {change_emoji} {acc_change_pct:+.2f}% ({format_korean_currency(acc_change)})\n"
            f"  - 현금: {format_korean_currency(acc['cash'])}"
        )
        acc_details.append(line)

    send_slack_message_v2("\n\n".join(acc_details), thread_ts=main_ts)

    # 3. Compose Portfolio Composition (Thread)
    if all_holdings:
        combined_df = pd.concat(all_holdings, ignore_index=True)
        bucket_cols = ["1. 모멘텀", "2. 혁신기술", "3. 시장지수", "4. 배당방어", "5. 대체헷지"]
        comp_details = ["*🏗️ 포트폴리오 구성 비중*"]

        for b in bucket_cols:
            b_val = combined_df.loc[combined_df["버킷"] == b, "평가금액(KRW)"].sum()
            b_pct = (b_val / total_assets * 100) if total_assets > 0 else 0.0
            comp_details.append(f"• {b}: {b_pct:.1f}% ({format_korean_currency(b_val)})")

        cash_pct = (global_cash / total_assets * 100) if total_assets > 0 else 0.0
        comp_details.append(f"• 6. 현금: {cash_pct:.1f}% ({format_korean_currency(global_cash)})")

        send_slack_message_v2("\n".join(comp_details), thread_ts=main_ts)

    # 4. Save Snapshots for next time (Consolidated)
    # Save individual accounts first, then TOTAL (which updates the same document for today)
    for acc in account_summaries:
        save_daily_snapshot(acc["account_id"], acc["total_assets"], acc["principal"], acc["cash"], acc["valuation"])

    save_daily_snapshot("TOTAL", total_assets, global_principal, global_cash, total_assets - global_cash)

    logger.info("Slack asset summary sent successfully.")


if __name__ == "__main__":
    main()
