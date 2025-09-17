"""
APScheduler 기반 스케줄러

환경변수로 크론 및 타임존을 설정할 수 있으며,
기본값은 각 국가 장 마감 이후(평일)로 설정됩니다.

ENV
- SCHEDULE_ENABLE_KOR/AUS/COIN: "1"/"0" (기본 1)
- SCHEDULE_KOR_CRON: 기본 "10 18 * * 1-5" (18:10, 평일)
- SCHEDULE_AUS_CRON: 기본 "10 18 * * 1-5"
- SCHEDULE_COIN_CRON: 기본 "5 0 * * *" (매일 00:05)
- SCHEDULE_KOR_TZ: 기본 "Asia/Seoul"
- SCHEDULE_AUS_TZ: 기본 "Australia/Sydney"
- SCHEDULE_COIN_TZ: 기본 "Asia/Seoul"
- RUN_IMMEDIATELY_ON_START: "1" 이면 시작 시 즉시 한 번 실행
"""

import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from utils.data_updater import update_etf_names

try:
    # DB에서 설정을 읽어 스케줄 주기를 제어
    from utils.db_manager import get_app_settings, get_common_settings
    from utils.env import load_env_if_present
except Exception:
    get_common_settings = lambda: None
    get_app_settings = lambda country: None
    load_env_if_present = lambda: False


def _bool_env(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip() not in ("0", "false", "False", "no", "NO")


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def run_status(country: str):
    """Run status generation and implicit Slack notification."""
    try:
        from status import main as run_status_main

        print(f"Running status for {country}")
        run_status_main(country=country, date_str=None)
    except Exception as e:
        print(f"Status job failed for {country}: {e}")


def _try_sync_bithumb_equity():
    """If coin: snapshot Bithumb balances into daily_equities before status run."""
    try:
        from scripts.snapshot_bithumb_balances import main as snapshot_main

        snapshot_main()
    except Exception as e:
        print(f"Bithumb balance snapshot skipped or failed: {e}")


def _try_sync_bithumb_trades():
    """If coin: sync Bithumb accounts → trades before status run."""
    try:
        from scripts.sync_bithumb_accounts_to_trades import main as sync_main

        sync_main()
    except Exception as e:
        print(f"Bithumb accounts→trades sync skipped or failed: {e}")


def main():
    # Load .env for API keys, DB, etc.
    load_env_if_present()

    # Update stock names before scheduling
    print("Checking for and updating stock names...")
    try:
        update_etf_names()
        print("Stock name update complete.")
    except Exception as e:
        print(f"Failed to update stock names: {e}")

    scheduler = BlockingScheduler()

    # 공통 설정에서 스케줄 주기(시간) 읽기
    common = get_common_settings() or {}

    # coin
    if _bool_env("SCHEDULE_ENABLE_COIN", True):
        # DB의 크론 설정을 우선 사용, 없으면 환경변수 폴백
        cron = common.get("SCHEDULE_CRON_COIN") or _get("SCHEDULE_COIN_CRON", "5 0 * * *")
        tz = _get("SCHEDULE_COIN_TZ", "Asia/Seoul")

        def coin_job():
            _try_sync_bithumb_trades()
            _try_sync_bithumb_equity()
            run_status("coin")

        scheduler.add_job(coin_job, CronTrigger.from_crontab(cron, timezone=tz), id="coin_status")
        print(f"Scheduled COIN: cron={cron} tz={tz}")

    # kor
    if _bool_env("SCHEDULE_ENABLE_KOR", True):
        cron = common.get("SCHEDULE_CRON_KOR") or _get("SCHEDULE_KOR_CRON", "10 18 * * 1-5")
        tz = _get("SCHEDULE_KOR_TZ", "Asia/Seoul")
        scheduler.add_job(
            run_status,
            CronTrigger.from_crontab(cron, timezone=tz),
            args=["kor"],
            id="kor_status",
        )
        print(f"Scheduled KOR: cron={cron} tz={tz}")

    # aus
    if _bool_env("SCHEDULE_ENABLE_AUS", True):
        cron = common.get("SCHEDULE_CRON_AUS") or _get("SCHEDULE_AUS_CRON", "10 18 * * 1-5")
        tz = _get("SCHEDULE_AUS_TZ", "Australia/Sydney")
        scheduler.add_job(
            run_status,
            CronTrigger.from_crontab(cron, timezone=tz),
            args=["aus"],
            id="aus_status",
        )
        print(f"Scheduled AUS: cron={cron} tz={tz}")

    if _bool_env("RUN_IMMEDIATELY_ON_START", False):
        # 시작 시 한 번 즉시 실행
        for c in ("coin", "kor", "aus"):
            try:
                if _bool_env(f"SCHEDULE_ENABLE_{c.upper()}", True):
                    run_status(c)
            except Exception:
                pass

    scheduler.start()


if __name__ == "__main__":
    main()
