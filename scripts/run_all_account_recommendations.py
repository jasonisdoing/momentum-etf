"""모든 계좌 추천을 순차 실행합니다."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.identifier_guard import ensure_account_pool_id_separation
from utils.logger import get_app_logger
from utils.settings_loader import list_available_accounts

logger = get_app_logger()


def main() -> int:
    load_env_if_present()
    ensure_account_pool_id_separation()

    account_ids = list_available_accounts()
    if not account_ids:
        logger.error("실행할 계좌가 없습니다.")
        return 1

    logger.info("전체 계좌 추천 실행 시작: %s", account_ids)

    failed: list[str] = []
    for index, account_id in enumerate(account_ids, start=1):
        logger.info(" -> 계좌 추천 실행 중: %d/%d - %s", index, len(account_ids), account_id)
        result = subprocess.run([sys.executable, "recommend.py", account_id], cwd=str(ROOT_DIR))
        if result.returncode != 0:
            failed.append(account_id)

    if failed:
        logger.error("실패한 계좌: %s", ", ".join(failed))
        return 1

    logger.info("전체 계좌 추천 실행 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
