from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from utils.account_registry import load_account_configs
from utils.env import load_env_if_present

load_env_if_present()

SystemAction = Literal["asset_summary"]

SCHEDULE_ROWS = [
    {
        "job": "가격 캐시 업데이트",
        "target": "모든 종목",
        "cadence": "매시 정각 KST",
        "command": "python scripts/stock_price_cache_updater.py",
    },
    {
        "job": "장 시간 분석",
        "target": "시장 스케줄",
        "cadence": "매일 07:00 KST",
        "command": "python scripts/analyze_market_hours.py",
    },
    {
        "job": "종목 메타데이터 업데이트",
        "target": "모든 종목타입",
        "cadence": "매일 09:00 KST",
        "command": "python scripts/stock_meta_cache_updater.py",
    },
    {
        "job": "전체 자산 요약 알림",
        "target": "전체 계좌",
        "cadence": "매일 09:30, 16:30 KST",
        "command": "python scripts/slack_asset_summary.py",
    },
]


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
    script_by_action = {
        "asset_summary": "scripts/slack_asset_summary.py",
    }
    message_by_action = {
        "asset_summary": "[시스템-정보] 전체 자산 요약 알림 전송 시작",
    }

    if action not in script_by_action:
        raise ValueError("지원하지 않는 시스템 작업입니다.")

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / script_by_action[action]
    env = os.environ.copy()
    subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=project_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return message_by_action[action]
