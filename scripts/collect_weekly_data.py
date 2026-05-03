from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가한다.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

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
