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
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

try:
    # DB에서 설정을 읽어 스케줄 주기를 제어
    from utils.db_manager import get_common_settings, get_app_settings
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
    """Run status generation and implicit Telegram notification."""
    try:
        from status import main as run_status_main
        logging.info("Running status for %s", country)
        run_status_main(country=country, date_str=None)
    except Exception as e:
        logging.exception("Status job failed for %s: %s", country, e)


def _try_sync_bithumb_equity():
    """If coin: snapshot Bithumb balances into daily_equities before status run."""
    try:
        from scripts.snapshot_bithumb_balances import main as snapshot_main
        snapshot_main()
    except Exception as e:
        logging.warning("Bithumb balance snapshot skipped or failed: %s", e)


def _try_sync_bithumb_trades():
    """If coin: sync Bithumb accounts → trades before status run."""
    try:
        from scripts.sync_bithumb_accounts_to_trades import main as sync_main
        sync_main()
    except Exception as e:
        logging.warning("Bithumb accounts→trades sync skipped or failed: %s", e)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

    # Load .env for API keys, DB, etc.
    load_env_if_present()

    scheduler = BlockingScheduler()

    # 공통 설정에서 스케줄 주기(시간) 읽기
    common = get_common_settings() or {}
    def _hours(name: str):
        try:
            v = common.get(name)
            return int(v) if v is not None and int(v) > 0 else None
        except Exception:
            return None

    # kor
    if _bool_env("SCHEDULE_ENABLE_KOR", True):
        every_h = _hours("SCHEDULE_EVERY_HOURS_KOR")
        if every_h:
            scheduler.add_job(run_status, IntervalTrigger(hours=every_h), args=["kor"], id="kor_status")
            logging.info("Scheduled KOR: every %sh", every_h)
        else:
            cron = _get("SCHEDULE_KOR_CRON", "10 18 * * 1-5")
            tz = _get("SCHEDULE_KOR_TZ", "Asia/Seoul")
            scheduler.add_job(run_status, CronTrigger.from_crontab(cron, timezone=tz), args=["kor"], id="kor_status")
            logging.info("Scheduled KOR: cron=%s tz=%s", cron, tz)

    # aus
    if _bool_env("SCHEDULE_ENABLE_AUS", True):
        every_h = _hours("SCHEDULE_EVERY_HOURS_AUS")
        if every_h:
            scheduler.add_job(run_status, IntervalTrigger(hours=every_h), args=["aus"], id="aus_status")
            logging.info("Scheduled AUS: every %sh", every_h)
        else:
            cron = _get("SCHEDULE_AUS_CRON", "10 18 * * 1-5")
            tz = _get("SCHEDULE_AUS_TZ", "Australia/Sydney")
            scheduler.add_job(run_status, CronTrigger.from_crontab(cron, timezone=tz), args=["aus"], id="aus_status")
            logging.info("Scheduled AUS: cron=%s tz=%s", cron, tz)

    # coin
    if _bool_env("SCHEDULE_ENABLE_COIN", True):
        every_h = _hours("SCHEDULE_EVERY_HOURS_COIN")
        if every_h:
            def coin_job():
                _try_sync_bithumb_trades()
                _try_sync_bithumb_equity()
                run_status("coin")
            scheduler.add_job(coin_job, IntervalTrigger(hours=every_h), id="coin_status")
            # scheduler.add_job(coin_job, IntervalTrigger(minutes=1), id="coin_status")
            logging.info("Scheduled COIN: every %sh", every_h)
        else:
            cron = _get("SCHEDULE_COIN_CRON", "5 0 * * *")
            tz = _get("SCHEDULE_COIN_TZ", "Asia/Seoul")
            def coin_job():
                _try_sync_bithumb_trades()
                _try_sync_bithumb_equity()
                run_status("coin")
            scheduler.add_job(coin_job, CronTrigger.from_crontab(cron, timezone=tz), id="coin_status")
            logging.info("Scheduled COIN: cron=%s tz=%s", cron, tz)

    if _bool_env("RUN_IMMEDIATELY_ON_START", False):
        # 시작 시 한 번 즉시 실행
        for c in ("kor", "aus", "coin"):
            try:
                if _bool_env(f"SCHEDULE_ENABLE_{c.upper()}", True):
                    run_status(c)
            except Exception:
                pass

    logging.info("Starting scheduler...")
    scheduler.start()


if __name__ == "__main__":
    main()
