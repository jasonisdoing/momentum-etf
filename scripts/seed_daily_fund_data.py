from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.daily_fund_service import seed_daily_data_from_weekly
from utils.env import load_env_if_present


def main() -> int:
    load_env_if_present()
    result = seed_daily_data_from_weekly()
    print(
        "일별 원장 시드 완료:",
        f"추가 {result['seeded']}건",
        f"건너뜀 {result['skipped']}건",
        f"주별 원본 {result['total_weekly_rows']}건",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
