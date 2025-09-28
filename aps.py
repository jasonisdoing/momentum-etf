"""
APScheduler 기반 스케줄러

[스케줄 설정]
스케줄은 아래 환경 변수를 통해 설정할 수 있습니다.
환경 변수가 없으면 각 작업의 기본값(Default)이 사용됩니다.

- SCHEDULE_ENABLE_KOR/AUS/COIN: "1" 또는 "0" (기본: "1", 활성화)
- SCHEDULE_KOR_CRON: 한국 시그널 계산 주기
- SCHEDULE_AUS_CRON: 호주 시그널 계산 주기
- SCHEDULE_COIN_CRON: 코인 시그널 계산 주기
- SCHEDULE_KOR_TZ: 한국 시간대 (기본: "Asia/Seoul")
- SCHEDULE_AUS_TZ: 호주 시간대 (기본: "Asia/Seoul")
- SCHEDULE_COIN_TZ: 코인 시간대 (기본: "Asia/Seoul")
- RUN_IMMEDIATELY_ON_START: "1" 이면 시작 시 즉시 한 번 실행 (기본: "0")
"""

import os
import logging
import sys
import warnings

# pkg_resources 워닝 억제
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import time
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from utils.data_updater import update_etf_names
from utils.env import load_env_if_present
from utils.account_registry import get_accounts_by_country, load_accounts
from utils.schedule_config import (
    get_all_country_schedules,
    get_cache_schedule,
    get_global_schedule_settings,
)


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


def _accounts_for_country(country: str) -> list[str]:
    try:
        load_accounts(force_reload=False)
        entries = get_accounts_by_country(country) or []
        accounts = []
        for entry in entries:
            code = entry.get("account")
            if code:
                accounts.append(str(code).strip())
        return accounts
    except Exception:
        logging.exception(f"Failed to load accounts for {country}")
        return []


def run_signal_generation(
    country: str,
    account: str | None = None,
    *,
    force_notify: bool = False,
) -> None:
    """Run signal generation and sends a completion log to Slack."""
    start_time = time.time()
    report_date = None
    try:
        from signals import main as run_signal_main
        from utils.notification import (
            send_summary_notification,
            send_detailed_signal_notification,
        )
        from utils.db_manager import get_portfolio_snapshot
        from utils.account_registry import get_account_info

        account_info = get_account_info(account) if account else None
        derived_country = (
            str(account_info.get("country") or "").strip() if account_info else country
        )
        snapshot_country = derived_country or country

        # 알림에 사용할 이전 평가금액을 미리 가져옵니다.
        old_snapshot = get_portfolio_snapshot(snapshot_country, account=account)
        old_equity = float(old_snapshot.get("total_equity", 0.0)) if old_snapshot else 0.0

        log_target = f"{snapshot_country}/{account}"
        logging.info(f"Running signal generation for {log_target}")
        if snapshot_country == "coin":
            _try_sync_bithumb_trades()

        # signal.main은 상세 알림을 처리합니다.
        signal_result = run_signal_main(account=account, date_str=None)

        # 작업이 성공적으로 완료되고 결과를 받아왔을 때만 요약 알림 전송
        if signal_result:
            report_date = signal_result.report_date
            duration = time.time() - start_time
            send_summary_notification(
                snapshot_country,
                account,
                report_date,
                duration,
                old_equity,
                summary_data=signal_result.summary_data,
                header_line=signal_result.header_line,
                force_send=force_notify,
            )
            time.sleep(2)
            send_detailed_signal_notification(
                snapshot_country,
                account,
                signal_result.header_line,
                signal_result.detail_headers,
                signal_result.detail_rows,
                decision_config=signal_result.decision_config,
                extra_lines=signal_result.detail_extra_lines,
                force_send=force_notify,
            )
            date_str = report_date.strftime("%Y-%m-%d")
            prefix = f"{snapshot_country}/{account}" if account else snapshot_country
            logging.info(f"[{prefix}/{date_str}] 작업 완료(작업시간: {duration:.1f}초)")

    except Exception:
        error_message = f"Signal generation job for {country}/{account} failed"
        logging.error(error_message, exc_info=True)


def run_signals_for_country(country: str, *, force_notify: bool = False) -> None:
    accounts = _accounts_for_country(country)
    if accounts:
        for account in accounts:
            try:
                run_signal_generation(country, account, force_notify=force_notify)
            except Exception:
                logging.error(
                    f"Error running signal generation for {country}/{account}", exc_info=True
                )
    else:
        logging.warning("No registered accounts for %s; skipping signal generation.", country)


def _try_sync_bithumb_trades():
    """If coin: sync Bithumb accounts → trades before status run."""
    try:
        from scripts.sync_bithumb_accounts_to_trades import main as sync_main

        sync_main()
    except Exception:
        error_message = "Bithumb accounts->trades sync skipped or failed"
        logging.error(error_message, exc_info=True)


def run_cache_refresh() -> None:
    """모든 국가의 가격 캐시를 갱신합니다."""
    start_date = os.environ.get("CACHE_START_DATE", "2020-01-01")
    countries_env = os.environ.get("CACHE_COUNTRIES", "kor,aus,coin")
    countries = [c.strip().lower() for c in countries_env.split(",") if c.strip()]
    logging.info("Running cache refresh (start=%s, countries=%s)", start_date, ",".join(countries))
    try:
        from scripts.update_price_cache import refresh_all_caches

        refresh_all_caches(countries=countries, start_date=start_date)
        logging.info("Cache refresh completed successfully")
    except Exception:
        logging.error("Cache refresh job failed", exc_info=True)


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
    # 1. signal_cron 와 notify_cron 이 겹치는 경우: 작업도 실행되고, 슬랙도 발송
    # 2. python cli.py --signal 로 실행되는 경우: 작업도 실행되고, 슬랙도 발송
    # 3. signal_cron 에는 해당되지만 notify_cron 에 해당 안되는 경우: 작업은 실행되고, 슬랙은 발송안됨
    cron_default = "0 0 * * *"
    tz_default = "Asia/Seoul"
    country_schedules = get_all_country_schedules()
    for country, cfg in country_schedules.items():
        enabled_default = bool(cfg.get("enabled", True))
        if not _bool_env(f"SCHEDULE_ENABLE_{country.upper()}", enabled_default):
            logging.info(f"Skipping {country.upper()} schedule (disabled)")
            continue

        cron_expr = _get(
            f"SCHEDULE_{country.upper()}_CRON",
            cfg.get("signal_cron", cron_default),
        )
        timezone = _get(
            f"SCHEDULE_{country.upper()}_TZ",
            cfg.get("timezone", tz_default),
        )

        scheduler.add_job(
            run_signals_for_country,
            CronTrigger.from_crontab(cron_expr, timezone=timezone),
            args=[country],
            id=country,
        )
        logging.info(f"Scheduled {country.upper()}: cron='{cron_expr}' tz='{timezone}'")

    cache_cfg = get_cache_schedule()
    cache_enabled_default = bool(cache_cfg.get("enabled", True))
    if _bool_env("SCHEDULE_ENABLE_CACHE", cache_enabled_default):
        cache_cron = _get("SCHEDULE_CACHE_CRON", cache_cfg.get("cron"))
        cache_tz = _get("SCHEDULE_CACHE_TZ", cache_cfg.get("timezone", tz_default))
        scheduler.add_job(
            run_cache_refresh,
            CronTrigger.from_crontab(cache_cron, timezone=cache_tz),
            id="price_cache_refresh",
        )
        logging.info(f"Scheduled CACHE: cron='{cache_cron}' tz='{cache_tz}'")

    global_schedule = get_global_schedule_settings()
    run_initial_default = bool(global_schedule.get("run_immediately_on_start", True))
    if _bool_env("RUN_IMMEDIATELY_ON_START", run_initial_default):
        logging.info("\n[Initial Run] Starting...")
        for country, cfg in country_schedules.items():
            try:
                enabled_default = bool(cfg.get("enabled", True))
                if _bool_env(f"SCHEDULE_ENABLE_{country.upper()}", enabled_default):
                    run_signals_for_country(country, force_notify=True)
            except Exception:
                logging.error(f"Error during initial run for {country}", exc_info=True)
        logging.info("[Initial Run] Complete.")
    else:
        logging.info("Initial run skipped (RUN_IMMEDIATELY_ON_START=0)")

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
