from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# 프로젝트 루트를 Python 경로에 추가한다.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.slack_asset_summary import (
    _load_latest_daily_metrics,
    _load_latest_monthly_metrics,
    _load_latest_weekly_metrics,
    _load_latest_yearly_metrics,
    collect_global_totals,
    format_korean_currency,
    get_trend_emoji,
)
from utils.daily_fund_service import aggregate_today_daily_data, remove_future_daily_rows
from utils.env import load_env_if_present
from utils.monthly_service import aggregate_active_month_data
from utils.notification import send_slack_message_v2
from utils.snapshot_service import update_today_snapshot_all_accounts
from utils.weekly_service import aggregate_active_week_data
from utils.yearly_service import aggregate_active_year_data

KST = ZoneInfo("Asia/Seoul")


def _send_data_aggregate_summary() -> None:
    daily_metrics = _load_latest_daily_metrics()
    weekly_metrics = _load_latest_weekly_metrics()
    monthly_metrics = _load_latest_monthly_metrics()
    yearly_metrics = _load_latest_yearly_metrics()
    totals = collect_global_totals()

    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    message = (
        f"*📊 데이터 집계 결과 ({now_str})*\n"
        f"💰 *총 자산*: *{format_korean_currency(totals['total_assets'])}*\n"
        f"📆 *금일 손익*: {format_korean_currency(daily_metrics['daily_profit'])} "
        f"({daily_metrics['daily_return_pct']:+.2f}%) {get_trend_emoji(daily_metrics['daily_profit'])}\n"
        f"💵 *현금 잔고*: {format_korean_currency(totals['global_cash'])} ({totals['cash_pct']:.1f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
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
    # 계좌별 금일 손익은 daily_snapshots, 합계는 daily_fund_data 를 소스로 쓴다.
    # 두 소스를 같은 실행·같은 시세로 맞추기 위해 집계 직전에 스냅샷을 먼저 갱신한다.
    # (스냅샷은 원래 보유 종목 변경 시에만 갱신돼, 매매 없는 날에는 기준이 낡을 수 있었다.)
    snapshot_result = update_today_snapshot_all_accounts()
    daily_result = aggregate_today_daily_data()
    weekly_result = aggregate_active_week_data()
    monthly_result = aggregate_active_month_data()
    yearly_result = aggregate_active_year_data()
    print(
        f"[data_aggregate] 데이터 집계 완료: "
        f"snapshot={snapshot_result['account_count']}계좌 "
        f"daily={daily_result['date']} "
        f"weekly={weekly_result['week_date']} "
        f"monthly={monthly_result['month_date']} "
        f"yearly={yearly_result['year_date']} "
        f"(미래 row 제거 {cleanup['deleted']}건)"
    )
    # 슬랙 발송 중단 (2026-08-03) — 이 배치는 월~토 매시 30분(주 144회) 도는데,
    # 보내던 내용(총자산·금일/금주/금월/금년 손익·현금)이 `전체 자산 요약`
    # (`scripts/slack_asset_summary.py`, 평일 09:40·16:10) 메시지에 그대로 포함된다.
    # 같은 내용이 14배 자주 와서 채널이 묻히므로 알림은 그쪽으로 일원화했다.
    # 집계 자체는 그대로 매시 수행한다. 배치가 죽는 경우는 `infra/cron/run_batch.py`
    # 래퍼가 여전히 `<!channel>` 로 알린다.
    # 되살리려면 아래 한 줄의 주석을 풀면 된다 (함수는 그대로 남겨 두었다).
    # _send_data_aggregate_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
