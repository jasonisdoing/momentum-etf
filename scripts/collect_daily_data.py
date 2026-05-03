from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.daily_fund_service import aggregate_today_daily_data, remove_future_daily_rows
from utils.env import load_env_if_present


def main() -> int:
    load_env_if_present()
    cleanup = remove_future_daily_rows()
    result = aggregate_today_daily_data()
    print(
        f"[daily_aggregate] 일별 데이터 집계 완료: {result['date']} "
        f"(미래 row 제거 {cleanup['deleted']}건)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
