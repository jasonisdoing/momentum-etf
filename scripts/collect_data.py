from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# 프로젝트 루트를 Python 경로에 추가한다.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.daily_fund_service import aggregate_today_daily_data, remove_future_daily_rows
from utils.env import load_env_if_present
from utils.monthly_service import aggregate_active_month_data
from utils.notification import send_slack_message_v2
from utils.weekly_service import aggregate_active_week_data
from utils.yearly_service import aggregate_active_year_data
from scripts.slack_asset_summary import (
    _load_latest_daily_metrics,
    _load_latest_monthly_metrics,
    _load_latest_weekly_metrics,
    _load_latest_yearly_metrics,
    format_korean_currency,
    get_trend_emoji,
)

KST = ZoneInfo("Asia/Seoul")


def _send_data_aggregate_summary() -> None:
    daily_metrics = _load_latest_daily_metrics()
    weekly_metrics = _load_latest_weekly_metrics()
    monthly_metrics = _load_latest_monthly_metrics()
    yearly_metrics = _load_latest_yearly_metrics()

    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    message = (
        f"*📊 데이터 집계 결과 ({now_str})*\n"
        f"📆 *금일 손익*: {format_korean_currency(daily_metrics['daily_profit'])} "
        f"({daily_metrics['daily_return_pct']:+.2f}%) {get_trend_emoji(daily_metrics['daily_profit'])}\n"
        f"🗓️ *금주 손익*: {format_korean_currency(weekly_metrics['weekly_profit'])} "
        f"({weekly_metrics['weekly_return_pct']:+.2f}%) {get_trend_emoji(weekly_metrics['weekly_profit'])}\n"
        f"🗓️ *금월 손익*: {format_korean_currency(monthly_metrics['monthly_profit'])} "
        f"({monthly_metrics['monthly_return_pct']:+.2f}%) {get_trend_emoji(monthly_metrics['monthly_profit'])}\n"
        f"📅 *금년 손익*: {format_korean_currency(yearly_metrics['yearly_profit'])} "
        f"({yearly_metrics['yearly_return_pct']:+.2f}%) {get_trend_emoji(yearly_metrics['yearly_profit'])}"
    )
    if not send_slack_message_v2(message):
        raise RuntimeError("데이터 집계 결과 슬랙 메시지 전송에 실패했습니다.")


def main() -> int:
    load_env_if_present()
    cleanup = remove_future_daily_rows()
    daily_result = aggregate_today_daily_data()
    weekly_result = aggregate_active_week_data()
    monthly_result = aggregate_active_month_data()
    yearly_result = aggregate_active_year_data()
    print(
        f"[data_aggregate] 데이터 집계 완료: "
        f"daily={daily_result['date']} "
        f"weekly={weekly_result['week_date']} "
        f"monthly={monthly_result['month_date']} "
        f"yearly={yearly_result['year_date']} "
        f"(미래 row 제거 {cleanup['deleted']}건)"
    )
    _send_data_aggregate_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
