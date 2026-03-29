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
        "job": "종목 메타데이터 업데이트",
        "target": "모든 계좌",
        "cadence": "매일 09:00 KST",
        "command": "python scripts/stock_meta_updater.py",
    },
    {
        "job": "가격 캐시 업데이트",
        "target": "모든 계좌",
        "cadence": "매시 정각 KST",
        "command": "python scripts/update_price_cache.py",
    },
    {
        "job": "전체 자산 요약 알림",
        "target": "전체 계좌 요약",
        "cadence": "매일 11:00, 18:00, 23:00, 06:00 KST",
        "command": "python scripts/slack_asset_summary.py",
    },
]


def load_system_data() -> dict[str, object]:
    accounts = load_account_configs()
    account_ids = [account["account_id"] for account in accounts]
    return {
        "summary_rows": [
            {
                "category": "계좌",
                "count": len(account_ids),
                "target": ", ".join(account_ids) if account_ids else "-",
            }
        ],
        "schedule_rows": SCHEDULE_ROWS,
        "schedule_note": "자동 주기는 현재 `.github/workflows` 기준입니다.",
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
