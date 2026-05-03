from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가한다.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.daily_fund_service import aggregate_today_daily_data, remove_future_daily_rows
from utils.env import load_env_if_present
from utils.monthly_service import aggregate_active_month_data
from utils.weekly_service import aggregate_active_week_data


def main() -> int:
    load_env_if_present()
    cleanup = remove_future_daily_rows()
    daily_result = aggregate_today_daily_data()
    weekly_result = aggregate_active_week_data()
    monthly_result = aggregate_active_month_data()
    print(
        f"[data_aggregate] 데이터 집계 완료: "
        f"daily={daily_result['date']} "
        f"weekly={weekly_result['week_date']} "
        f"monthly={monthly_result['month_date']} "
        f"(미래 row 제거 {cleanup['deleted']}건)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
