from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from utils.account_registry import load_account_configs
from utils.env import load_env_if_present

load_env_if_present()

SystemAction = Literal[
    "cache_refresh",
    "market_hours_analysis",
    "metadata_updater",
    "asset_summary",
]

# 배치 정의: 키는 infra/cron/crontab 의 job name 과 동일해야 합니다.
SCHEDULE_ROWS = [
    {
        "key": "cache_refresh",
        "job": "가격 캐시 업데이트",
        "target": "모든 종목",
        "cadence": "매시 정각 KST",
        "command": "python scripts/stock_price_cache_updater.py",
    },
    {
        "key": "market_hours_analysis",
        "job": "장 시간 분석",
        "target": "시장 스케줄",
        "cadence": "매일 07:00 KST",
        "command": "python scripts/analyze_market_hours.py",
    },
    {
        "key": "metadata_updater",
        "job": "종목 메타데이터 업데이트",
        "target": "모든 종목타입",
        "cadence": "매일 09:00 KST",
        "command": "python scripts/stock_meta_cache_updater.py",
    },
    {
        "key": "asset_summary",
        "job": "전체 자산 요약 알림",
        "target": "전체 계좌",
        "cadence": "매일 09:30, 16:30 KST",
        "command": "python scripts/slack_asset_summary.py",
    },
]

# action 키 → 실행할 스크립트 경로
_SCRIPT_BY_ACTION: dict[str, str] = {
    "cache_refresh": "scripts/stock_price_cache_updater.py",
    "market_hours_analysis": "scripts/analyze_market_hours.py",
    "metadata_updater": "scripts/stock_meta_cache_updater.py",
    "asset_summary": "scripts/slack_asset_summary.py",
}

_LABEL_BY_ACTION: dict[str, str] = {
    row["key"]: row["job"] for row in SCHEDULE_ROWS
}


def load_system_data() -> dict[str, object]:
    accounts = load_account_configs()
    return {
        "summary_rows": [
            {
                "category": f"{account.get('icon', '')} {account['name']}".strip(),
                "count": int(account["order"]),
                "target": account["account_id"],
            }
            for account in accounts
        ],
        "schedule_rows": SCHEDULE_ROWS,
        "schedule_note": (
            "VM 호스트 cron 이 `infra/cron/run_batch.py` 래퍼를 통해 실행하며 "
            "성공/실패 결과를 슬랙으로 알립니다. (정의: `infra/cron/crontab`)"
        ),
    }


def trigger_system_action(action: SystemAction) -> str:
    """배치를 백그라운드로 실행. cron 과 동일하게 run_batch.py 래퍼를 경유해
    실행 결과를 슬랙으로 알립니다."""

    if action not in _SCRIPT_BY_ACTION:
        raise ValueError("지원하지 않는 시스템 작업입니다.")

    project_root = Path(__file__).resolve().parents[1]
    script_rel = _SCRIPT_BY_ACTION[action]
    wrapper_rel = "infra/cron/run_batch.py"

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    subprocess.Popen(
        [sys.executable, wrapper_rel, action, sys.executable, script_rel],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    label = _LABEL_BY_ACTION.get(action, action)
    return f"[시스템-배치] {label} 백그라운드 실행을 시작했습니다."
