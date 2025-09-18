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
import logging
import sys
import time
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.data_updater import update_etf_names

try:
    # DB에서 설정을 읽어 스케줄 주기를 제어
    from utils.db_manager import get_app_settings, get_common_settings
    from utils.env import load_env_if_present
except Exception:

    def get_common_settings():
        return None

    def get_app_settings(country):
        return None

    def load_env_if_present():
        return False


def setup_logging():
    """
    로그 파일을 설정합니다. logs/YYYY-MM-DD.log 형식으로 생성됩니다.
    프로세스가 시작될 때의 날짜를 기준으로 파일명이 정해집니다.
    """
    # 프로젝트 루트 아래에 logs 디렉토리 생성
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # YYYY-MM-DD.log 파일명 설정
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

    # 로거 설정: 파일과 콘솔에 모두 출력
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _bool_env(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip() not in ("0", "false", "False", "no", "NO")


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _format_korean_datetime(dt: datetime) -> str:
    """날짜-시간 객체를 'YYYY년 MM월 DD일(요일) 오전/오후 HH시 MM분' 형식으로 변환합니다."""
    weekday_map = ["월", "화", "수", "목", "금", "토", "일"]
    weekday_str = weekday_map[dt.weekday()]

    hour12 = dt.hour
    if hour12 >= 12:
        ampm_str = "오후"
        if hour12 > 12:
            hour12 -= 12
    else:
        ampm_str = "오전"
    if hour12 == 0:
        hour12 = 12

    return f"{dt.strftime('%Y년 %m월 %d일')}({weekday_str}) {ampm_str} {hour12}시 {dt.minute:02d}분"


def run_status(country: str) -> None:
    """Run status generation and sends a completion log to Slack."""
    start_time = time.time()
    report_date = None
    try:
        from status import main as run_status_main
        from utils.notify import send_log_to_slack
        from utils.db_manager import get_portfolio_snapshot, get_app_settings
        from utils.report import format_aud_money, format_kr_money

        # Get old equity
        old_snapshot = get_portfolio_snapshot(country)
        old_equity = float(old_snapshot.get("total_equity", 0.0)) if old_snapshot else 0.0

        logging.info(f"Running status for {country}")
        if country == "coin":
            _try_sync_bithumb_trades()
            # _try_sync_bithumb_equity()

        # status.main은 성공 시 계산된 리포트의 기준 날짜를 반환합니다.
        report_date = run_status_main(country=country, date_str=None)

        # 작업이 성공적으로 완료되고 날짜를 받아왔을 때만 로그 전송
        if report_date:
            duration = time.time() - start_time
            date_str = report_date.strftime("%Y-%m-%d")
            message = f"{country}/{date_str} 작업 완료(작업시간: {duration:.1f}초)"

            # Get new equity
            new_snapshot = get_portfolio_snapshot(country)
            new_equity = float(new_snapshot.get("total_equity", 0.0)) if new_snapshot else 0.0

            # Calculate cumulative return
            app_settings = get_app_settings(country)
            initial_capital = float(app_settings.get("initial_capital", 0)) if app_settings else 0.0

            money_formatter = format_aud_money if country == "aus" else format_kr_money

            if initial_capital > 0:
                cum_ret_pct = ((new_equity / initial_capital) - 1.0) * 100.0
                cum_profit_loss = new_equity - initial_capital
                equity_summary = f"평가금액: {money_formatter(new_equity)}, 누적수익 {cum_ret_pct:+.2f}%({money_formatter(cum_profit_loss)})"
                message += f" | {equity_summary}"

            if abs(new_equity - old_equity) > 1e-9:
                diff = new_equity - old_equity
                diff_str = f"{'+' if diff > 0 else ''}{money_formatter(diff)}"
                change_label = "📈평가금액 증가" if diff >= 0 else "📉평가금액 감소"
                equity_change_message = f"{change_label}: {money_formatter(old_equity)} => {money_formatter(new_equity)} ({diff_str})"
                message += f" | {equity_change_message}"

            send_log_to_slack(message)

    except Exception:
        error_message = f"Status job for {country} failed"
        logging.error(error_message, exc_info=True)


def _try_sync_bithumb_equity():
    """
    코인(Bithumb) 잔액을 스냅샷하고, 변경된 경우 슬랙으로 알림을 보냅니다.
    """
    try:
        from scripts.snapshot_bithumb_balances import main as snapshot_main
        from status import _notify_equity_update
        from utils.db_manager import get_portfolio_snapshot, save_daily_equity

        # 1. 업데이트 전 현재 평가금액을 가져옵니다.
        old_snapshot = get_portfolio_snapshot("coin")
        old_equity = float(old_snapshot.get("total_equity", 0.0)) if old_snapshot else 0.0

        # 2. 빗썸 잔액 스냅샷 스크립트를 실행하여 DB를 업데이트합니다.
        logging.info("Starting Bithumb balance snapshot...")
        snapshot_main()
        logging.info("Bithumb balance snapshot finished.")

        # 3. 업데이트 후 새로운 평가금액을 가져옵니다.
        new_snapshot = get_portfolio_snapshot("coin")
        new_equity = float(new_snapshot.get("total_equity", 0.0)) if new_snapshot else 0.0

        # 4. 스케줄러에 의한 업데이트임을 기록하기 위해 `updated_by`와 함께 항상 저장합니다.
        if new_snapshot:
            save_daily_equity("coin", new_snapshot["date"], new_equity, updated_by="스케줄러")
            logging.info("-> Coin equity snapshot updated. (updated_by='scheduler')")

            # 5. 평가금액이 변경되었는지 확인하고, 변경된 경우 슬랙 알림을 보냅니다.
            if abs(new_equity - old_equity) > 1e-9:
                logging.info(
                    f"-> Coin equity change detected: {old_equity:,.0f} -> {new_equity:,.0f}. Sending notification."
                )
                _notify_equity_update("coin", old_equity, new_equity)
            else:
                logging.info("-> No change in coin equity.")
        else:
            logging.warning("-> Coin equity snapshot not found, skipping update.")

    except Exception:
        error_message = "Bithumb balance snapshot skipped or failed"
        logging.error(error_message, exc_info=True)


def _try_sync_bithumb_trades():
    """If coin: sync Bithumb accounts → trades before status run."""
    try:
        from scripts.sync_bithumb_accounts_to_trades import main as sync_main

        sync_main()
    except Exception:
        error_message = "Bithumb accounts->trades sync skipped or failed"
        logging.error(error_message, exc_info=True)


def main():
    # 로깅 설정
    setup_logging()

    # Load .env for API keys, DB, etc.
    load_env_if_present()

    # Update stock names before scheduling
    logging.info("Checking for and updating stock names...")
    try:
        update_etf_names()
        logging.info("Stock name update complete.")
    except Exception as e:
        logging.error(f"Failed to update stock names: {e}", exc_info=True)

    scheduler = BlockingScheduler()

    # 공통 설정에서 스케줄 주기(시간) 읽기
    common = get_common_settings() or {}
    # coin
    if _bool_env("SCHEDULE_ENABLE_COIN", True):
        # DB의 크론 설정을 우선 사용, 없으면 환경변수 폴백
        cron = common.get("SCHEDULE_CRON_COIN") or _get("SCHEDULE_COIN_CRON", "5 0 * * *")
        tz = _get("SCHEDULE_COIN_TZ", "Asia/Seoul")
        scheduler.add_job(
            run_status,
            CronTrigger.from_crontab(cron, timezone=tz),
            args=["coin"],
            id="coin",
        )
        logging.info(f"Scheduled COIN: cron='{cron}' tz='{tz}'")

    # aus
    if _bool_env("SCHEDULE_ENABLE_AUS", True):
        cron = common.get("SCHEDULE_CRON_AUS") or _get("SCHEDULE_AUS_CRON", "10 18 * * 1-5")
        tz = _get("SCHEDULE_AUS_TZ", "Australia/Sydney")
        scheduler.add_job(
            run_status,
            CronTrigger.from_crontab(cron, timezone=tz),
            args=["aus"],
            id="aus",
        )
        logging.info(f"Scheduled AUS: cron='{cron}' tz='{tz}'")

    # kor
    if _bool_env("SCHEDULE_ENABLE_KOR", True):
        cron = common.get("SCHEDULE_CRON_KOR") or _get("SCHEDULE_KOR_CRON", "10 18 * * 1-5")
        tz = _get("SCHEDULE_KOR_TZ", "Asia/Seoul")
        scheduler.add_job(
            run_status,
            CronTrigger.from_crontab(cron, timezone=tz),
            args=["kor"],
            id="kor",
        )
        logging.info(f"Scheduled KOR: cron='{cron}' tz='{tz}'")

    if _bool_env("RUN_IMMEDIATELY_ON_START", False):
        # 시작 시 한 번 즉시 실행
        logging.info("\n[Initial Run] Starting...")
        # run_status("aus")
        for country in ("coin", "aus", "kor"):
            try:
                if _bool_env(f"SCHEDULE_ENABLE_{country.upper()}", True):
                    run_status(country)
            except Exception:
                logging.error(f"Error during initial run for {country}", exc_info=True)
        logging.info("[Initial Run] Complete.")

    # 다음 실행 시간 출력
    jobs = scheduler.get_jobs()
    if jobs:
        logging.info("\nNext scheduled run times:")
        for job in jobs:
            # 3.x: job.next_run_time
            next_time = getattr(job, "next_run_time", None)

            # 4.x (혹시 모르게 혼합된 경우): trigger에서 직접 구하기
            if next_time is None and hasattr(job, "trigger"):
                next_time = job.trigger.get_next_fire_time(None, datetime.now())

            if next_time:
                logging.info(f"- {job.id}: {_format_korean_datetime(next_time)}")
            else:
                logging.info(f"- {job.id}: No scheduled runs")
    logging.info("\nStarting scheduler. Waiting for the next job...")
    scheduler.start()


if __name__ == "__main__":
    main()
