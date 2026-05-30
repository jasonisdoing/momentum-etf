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
from zoneinfo import ZoneInfo

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.account_registry import load_account_configs
from utils.daily_fund_service import load_daily_docs_for_aggregation
from utils.db_manager import get_db_connection
from utils.env import load_env_if_present
from utils.notification import send_slack_message_v2
from utils.portfolio_io import (
    MissingPriceCacheError,
    get_latest_daily_snapshot,
    load_portfolio_master,
    load_real_holdings_table,
    save_daily_snapshot,
)
from utils.report import format_kr_money, format_kr_money_man
from utils.monthly_service import (
    MONTHLY_COLLECTION,
    _apply_running_total_principal as _apply_monthly_running,
)
from utils.weekly_service import _apply_running_total_principal as _apply_weekly_running
from utils.yearly_service import (
    YEARLY_COLLECTION,
    _apply_running_total_principal as _apply_yearly_running,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# 슬랙 알림(데이터 집계 결과 / 총 자산 요약)에서 사용하는 금액 포맷.
# 만원 미만은 절사하여 간결하게 표시한다.
format_korean_currency = format_kr_money_man
# format_kr_money 는 다른 곳(계좌별 상세 등)에서 필요할 수 있어 import 만 유지.
_ = format_kr_money
WEEKLY_COLLECTION = "weekly_fund_data"
INITIAL_TOTAL_PRINCIPAL_DATE = "2024-01-31"
INITIAL_TOTAL_PRINCIPAL_VALUE = 56_000_000
KST = ZoneInfo("Asia/Seoul")


def get_trend_emoji(val):
    if val > 0:
        return ":small_red_triangle:"
    elif val < 0:
        return ":chart_with_downwards_trend:"
    return ""


def _to_int(value):
    return int(value or 0)


def _calculate_total_expense(doc):
    return (
        _to_int(doc.get("withdrawal_personal", 0))
        + _to_int(doc.get("withdrawal_mom", 0))
        + _to_int(doc.get("nh_principal_interest", 0))
    )


def _load_latest_daily_metrics():
    """일별 데이터 테이블과 동일한 정의(TWR 1일)로 최신 일간 손익 지표를 반환한다."""
    docs = load_daily_docs_for_aggregation()
    if not docs:
        raise RuntimeError("daily_fund_data 데이터가 없어 일간 손익을 계산할 수 없습니다.")
    latest = docs[0]  # _apply_running_total_principal 결과는 date 내림차순
    return {
        "date": str(latest.get("date") or "").strip(),
        "daily_profit": _to_int(latest.get("daily_profit", 0)),
        "daily_return_pct": float(latest.get("daily_return_pct", 0.0) or 0.0),
    }


def _load_latest_weekly_metrics():
    """주별 데이터 테이블과 동일한 정의(TWR 1주, ROI 누적)로 최신 주간 손익 지표를 반환한다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패로 주별 데이터를 조회할 수 없습니다.")

    docs = list(db[WEEKLY_COLLECTION].find().sort("week_date", 1))
    if not docs:
        raise RuntimeError("weekly_fund_data 데이터가 없어 주간 손익을 계산할 수 없습니다.")

    enriched = _apply_weekly_running(docs)  # 결과는 week_date 내림차순
    if not enriched:
        raise RuntimeError("주간 손익 계산 결과를 만들지 못했습니다.")
    latest = enriched[0]
    return {
        "week_date": str(latest.get("week_date") or "").strip(),
        "weekly_profit": _to_int(latest.get("weekly_profit", 0)),
        "weekly_return_pct": float(latest.get("weekly_return_pct", 0.0) or 0.0),
        "cumulative_profit": _to_int(latest.get("cumulative_profit", 0)),
        "cumulative_return_pct": float(latest.get("cumulative_return_pct", 0.0) or 0.0),
    }


def _load_latest_monthly_metrics():
    """월별 데이터 테이블과 동일한 정의(TWR 1월)로 최신 월간 손익 지표를 반환한다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패로 월별 데이터를 조회할 수 없습니다.")

    docs = list(db[MONTHLY_COLLECTION].find().sort("month_date", 1))
    if not docs:
        raise RuntimeError("monthly_fund_data 데이터가 없어 월간 손익을 계산할 수 없습니다.")

    enriched = _apply_monthly_running(docs)  # 결과는 month_date 내림차순
    if not enriched:
        raise RuntimeError("월간 손익 계산 결과를 만들지 못했습니다.")
    latest = enriched[0]
    return {
        "month_date": str(latest.get("month_date") or "").strip(),
        "monthly_profit": _to_int(latest.get("monthly_profit", 0)),
        "monthly_return_pct": float(latest.get("monthly_return_pct", 0.0) or 0.0),
    }


def _load_latest_yearly_metrics():
    """연별 데이터 테이블과 동일한 정의(TWR 1년)로 최신 연간 손익 지표를 반환한다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패로 연별 데이터를 조회할 수 없습니다.")

    docs = list(db[YEARLY_COLLECTION].find().sort("year_date", 1))
    if not docs:
        raise RuntimeError("yearly_fund_data 데이터가 없어 연간 손익을 계산할 수 없습니다.")

    enriched = _apply_yearly_running(docs)  # 결과는 year_date 내림차순
    if not enriched:
        raise RuntimeError("연간 손익 계산 결과를 만들지 못했습니다.")
    latest = enriched[0]
    return {
        "year_date": str(latest.get("year_date") or "").strip(),
        "yearly_profit": _to_int(latest.get("yearly_profit", 0)),
        "yearly_return_pct": float(latest.get("yearly_return_pct", 0.0) or 0.0),
    }


def _build_missing_cache_alert(account_id: str, tickers: list[str]) -> str:
    ticker_text = ", ".join(tickers)
    return (
        f"⚠️ 자산 요약 발송 중단 ({account_id})\n"
        f"가격 캐시가 없는 보유 종목: {ticker_text}\n"
        f"종목 관리에서 해당 종목의 메타/캐시 새로고침을 실행하세요."
    )


def collect_global_totals() -> dict[str, object]:
    """전체 계좌 합산 — 총 자산 / 투자 원금 / 현금 잔고 / 자산 변동 텍스트 반환.

    슬랙의 총 자산 요약과 데이터 집계 결과 두 메시지가 공통으로 사용한다.
    포맷팅된 텍스트(asset_change_text)는 슬랙용 이모지 형식이다.
    """
    accounts = load_account_configs()
    global_principal = 0.0
    global_cash = 0.0
    total_assets = 0.0
    for account in accounts:
        account_id = account["account_id"]
        if not account.get("settings", {}).get("show_hold", True):
            continue
        m_data = load_portfolio_master(account_id) or {}
        principal = float(m_data.get("total_principal") or 0.0)
        cash = float(m_data.get("cash_balance") or 0.0)
        global_principal += principal
        global_cash += cash
        try:
            df = load_real_holdings_table(account_id, strict_price_cache=False)
        except Exception:
            df = None
        valuation = float(df["평가금액(KRW)"].sum()) if df is not None and not df.empty else 0.0
        total_assets += valuation + cash

    cash_pct = (global_cash / total_assets * 100.0) if total_assets > 0 else 0.0

    # 전일 대비 자산 변동
    prev_total_snapshot = get_latest_daily_snapshot("TOTAL", before_today=True)
    asset_change_text = ""
    if prev_total_snapshot:
        prev_total_assets = float(prev_total_snapshot.get("total_assets") or 0.0)
        if prev_total_assets > 0:
            delta = total_assets - prev_total_assets
            if delta > 0:
                asset_change_text = f" ({format_korean_currency(delta)}:small_red_triangle:)"
            elif delta < 0:
                asset_change_text = f" ({format_korean_currency(delta)}:chart_with_downwards_trend:)"

    return {
        "total_assets": total_assets,
        "global_principal": global_principal,
        "global_cash": global_cash,
        "cash_pct": cash_pct,
        "asset_change_text": asset_change_text,
    }


def main():
    load_env_if_present()

    accounts = load_account_configs()
    if not accounts:
        logger.error("No account configurations found.")
        return

    all_holdings_dict = {}
    account_summaries = []
    global_principal = 0.0
    global_cash = 0.0
    total_purchase = 0.0

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
            # 일부 공용 유틸이 환경 설정을 필요로 할 수 있어 예외를 명확히 처리한다.
            df = load_real_holdings_table(account_id, strict_price_cache=True)
        except MissingPriceCacheError as e:
            alert_msg = _build_missing_cache_alert(account_id, e.tickers)
            logger.error(alert_msg)
            send_slack_message_v2(alert_msg)
            sys.exit(1)
        except Exception as e:
            error_msg = f"❌ 자산 요약 생성 중 치명적 에러 발생 ({account_id} 계좌):\n```{e}```\n\n잘못된 자산 리포트가 발송되는 것을 방지하기 위해 오늘 알림을 중단합니다."
            logger.error(error_msg)
            send_slack_message_v2(error_msg)
            sys.exit(1)

        acc_valuation = 0.0
        acc_purchase = 0.0
        if df is not None and not df.empty:
            all_holdings_dict[account_id] = df
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
            total_purchase += acc_purchase

    if not account_summaries:
        logger.warning("No data found to report.")
        return

    # Global Calculations
    total_assets = sum(acc["total_assets"] for acc in account_summaries)

    # 일/주/누적 손익은 /daily, /weekly 화면과 동일한 값으로 통일.
    # (자산 수익률 계산 정책: docs/developer_guide.md)
    daily_metrics = _load_latest_daily_metrics()
    weekly_metrics = _load_latest_weekly_metrics()
    monthly_metrics = _load_latest_monthly_metrics()
    yearly_metrics = _load_latest_yearly_metrics()
    cash_pct = (global_cash / total_assets * 100) if total_assets > 0 else 0.0

    # 1. Compose Main Message (Total Summary)
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

    main_text = (
        f"*📊 총 자산 요약 ({now_str})*\n"
        f"💰 *총 자산*: *{format_korean_currency(total_assets)}*\n"
        f"📆 *금일 손익*: {format_korean_currency(daily_metrics['daily_profit'])} "
        f"({daily_metrics['daily_return_pct']:+.2f}%) {get_trend_emoji(daily_metrics['daily_profit'])}\n"
    )

    main_ts = send_slack_message_v2(main_text)
    if not main_ts:
        logger.error("Failed to send main Slack message.")
        return

    # 2. Compose Profit & Loss Summary (Thread) — 금일/금주/금월/금년/누적 + 현금 잔고
    pnl_text = (
        f"📆 *금일 손익*: {format_korean_currency(daily_metrics['daily_profit'])} ({daily_metrics['daily_return_pct']:+.2f}%) {get_trend_emoji(daily_metrics['daily_profit'])}\n"
        f"🗓️ *금주 손익*: {format_korean_currency(weekly_metrics['weekly_profit'])} ({weekly_metrics['weekly_return_pct']:+.2f}%) {get_trend_emoji(weekly_metrics['weekly_profit'])}\n"
        f"🗓️ *금월 손익*: {format_korean_currency(monthly_metrics['monthly_profit'])} ({monthly_metrics['monthly_return_pct']:+.2f}%) {get_trend_emoji(monthly_metrics['monthly_profit'])}\n"
        f"📅 *금년 손익*: {format_korean_currency(yearly_metrics['yearly_profit'])} ({yearly_metrics['yearly_return_pct']:+.2f}%) {get_trend_emoji(yearly_metrics['yearly_profit'])}\n"
        f"🏁 *누적 손익*: *{format_korean_currency(weekly_metrics['cumulative_profit'])} ({weekly_metrics['cumulative_return_pct']:+.2f}%)* {get_trend_emoji(weekly_metrics['cumulative_profit'])}\n"
        f"💵 *현금 잔고*: {format_korean_currency(global_cash)} ({cash_pct:.1f}%)"
    )
    send_slack_message_v2(pnl_text, thread_ts=main_ts)

    # 3. Compose Account Details (Thread)
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
        acc_cash_pct = (acc["cash"] / acc["total_assets"] * 100) if acc["total_assets"] > 0 else 0.0
        line = (
            f"• *{acc['name']}*\n"
            f"  - 자산: {format_korean_currency(acc['total_assets'])} (원금: {format_korean_currency(acc['principal'])})\n"
            f"  - 누적수익: {emoji} {acc['net_profit_pct']:+.2f}% ({format_korean_currency(acc['net_profit'])})\n"
            f"  - 금일변동: {change_emoji} {acc_change_pct:+.2f}% ({format_korean_currency(acc_change)})\n"
            f"  - 현금: {format_korean_currency(acc['cash'])} ({acc_cash_pct:.1f}%)"
        )
        acc_details.append(line)

    send_slack_message_v2("\n\n".join(acc_details), thread_ts=main_ts)

    # 3. Compose Portfolio Composition (Thread)
    if all_holdings_dict:
        combined_df = pd.concat(all_holdings_dict.values(), ignore_index=True)
        bucket_cols = ["1. 모멘텀", "2. 시장지수", "3. 배당방어", "4. 대체헷지"]
        comp_details = ["*🏗️ 포트폴리오 구성 비중*"]

        for b in bucket_cols:
            b_val = combined_df.loc[combined_df["버킷"] == b, "평가금액(KRW)"].sum()
            b_pct = (b_val / total_assets * 100) if total_assets > 0 else 0.0
            comp_details.append(f"• {b}: {b_pct:.1f}%")

        cash_pct = (global_cash / total_assets * 100) if total_assets > 0 else 0.0
        comp_details.append(f"• 5. 현금: {cash_pct:.1f}%")

        send_slack_message_v2("\n".join(comp_details), thread_ts=main_ts)

    # 4. Compose Account Cash Ratio (Thread)
    cash_ratio_details = ["*💵 계좌별 현금 비율*"]
    for acc in account_summaries:
        acc_cash_pct = (acc["cash"] / acc["total_assets"] * 100) if acc["total_assets"] > 0 else 0.0
        cash_ratio_details.append(f"• {acc['name']}: {acc_cash_pct:.1f}%")

    send_slack_message_v2("\n".join(cash_ratio_details), thread_ts=main_ts)

    # 5. Save Snapshots for next time (Consolidated)
    # Save individual accounts first, then TOTAL (which updates the same document for today)
    for acc in account_summaries:
        # 계좌별 실보유 종목 추출 (티커, 보유주수 정수값)
        holding_list = []
        target_df = all_holdings_dict.get(acc["account_id"])
        if target_df is not None:
            for _, row in target_df.iterrows():
                ticker = str(row.get("티커", "")).strip().upper()
                if ticker and ticker != "IS": # International Shares 등 가상 종목 제외
                    holding_list.append({"ticker": ticker})

        save_daily_snapshot(
            acc["account_id"],
            acc["total_assets"],
            acc["principal"],
            acc["cash"],
            acc["valuation"],
            acc.get("valuation", 0.0) - acc.get("stock_profit", 0.0),
            holding_details=holding_list
        )

    save_daily_snapshot(
        "TOTAL", total_assets, global_principal, global_cash, total_assets - global_cash, total_purchase
    )

    logger.info("Slack asset summary sent successfully.")


if __name__ == "__main__":
    main()
