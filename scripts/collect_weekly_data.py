from __future__ import annotations

from utils.env import load_env_if_present
from utils.weekly_service import aggregate_active_week_data


def main() -> int:
    load_env_if_present()
    result = aggregate_active_week_data()
    week_date = str(result.get("week_date", "")).strip()
    print(f"[weekly_aggregate] 주별 데이터 집계 완료: {week_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
