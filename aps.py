"""
APScheduler 기반 스케줄러

[스케줄 설정]
<data/settings/account/<account_id>.json>
"""

import logging
import os
import sys
import warnings

TIMEZONE = "Asia/Seoul"

# pkg_resources 워닝 억제
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import time
from datetime import datetime

from utils.recommendation_storage import save_recommendation_report


try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

try:  # pragma: no cover - 선택적 의존성 처리
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from logic.recommend.pipeline import (
    RecommendationReport,
    generate_account_recommendation_report,
)
from utils.env import load_env_if_present
from utils.notification import (
    compose_recommendation_slack_message,
    send_recommendation_slack_notification,
    should_notify_on_schedule,
)
from utils.schedule_config import get_all_country_schedules
from utils.data_loader import is_trading_day
from utils.cron_utils import normalize_cron_weekdays


def setup_logging():
    """
    로그 파일을 설정합니다. logs/YYYY-MM-DD.log 형식으로 생성됩니다.
    프로세스가 시작될 때의 날짜를 기준으로 파일명이 정해집니다.
    """
    # 프로젝트 루트 아래에 logs 디렉토리 생성
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # YYYY-MM-DD.log 파일명 설정
    if ZoneInfo is not None:
        try:
            now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        except Exception:
            now_kst = datetime.now()
    else:
        now_kst = datetime.now()

    log_filename = os.path.join(log_dir, f"{now_kst.strftime('%Y-%m-%d')}.log")

    # 로거 설정: 파일과 콘솔에 모두 출력
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


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

    return f"{dt.strftime('%Y년 %m월 %d일')}(" f"{weekday_str}) {ampm_str} {hour12}시 {dt.minute:02d}분"


def run_recommendation_generation(
    account_id: str,
    *,
    country_code: str,
    schedule_timezone: str | None = None,
) -> RecommendationReport:
    """Run portfolio recommendation and optionally notify Slack."""

    start_ts = time.time()
    logging.info("[APS] 추천 생성 시작: account=%s country=%s", account_id.upper(), country_code.upper())

    try:
        report = generate_account_recommendation_report(account_id=account_id)
    except Exception:
        logging.error("추천 데이터 생성에 실패했습니다.", exc_info=True)
        raise

    if not isinstance(report, RecommendationReport):
        raise TypeError("예상한 RecommendationReport 타입이 아닙니다.")

    elapsed = time.time() - start_ts
    logging.info(
        "[APS] 추천 생성 완료(account=%s, country=%s, elapsed=%.2fs)",
        account_id.upper(),
        country_code.upper(),
        elapsed,
    )

    logging.info("[APS] 보고서 저장 중...")
    try:
        save_recommendation_report(report)
    except Exception:
        logging.error("추천 보고서를 저장하는 중 오류", exc_info=True)

    try:
        if should_notify_on_schedule(country_code):
            message = compose_recommendation_slack_message(
                account_id,
                report,
                duration=elapsed,
            )
            send_recommendation_slack_notification(account_id, message)
    except Exception:
        logging.error("Slack 알림 전송 실패", exc_info=True)

    return report


def run_cache_refresh() -> None:
    """모든 국가의 가격 캐시를 갱신합니다."""
    from utils.account_registry import get_common_file_settings

    common_settings = get_common_file_settings()
    start_date = str(common_settings.get("CACHE_START_DATE") or "2020-01-01")
    countries = ["kor", "aus"]
    logging.info("Running cache refresh (start=%s, countries=%s)", start_date, ",".join(countries))
    try:
        from scripts.update_price_cache import refresh_all_caches

        refresh_all_caches(countries=countries, start_date=start_date)
        logging.info("Cache refresh completed successfully")
    except Exception:
        logging.error("가격 캐시 갱신 작업이 실패했습니다.", exc_info=True)


def run_stock_stats_update() -> None:
    """종목 파일의 메타데이터(상장일, 거래량 등)를 갱신합니다."""
    logging.info("Running stock metadata update...")
    try:
        from utils.stock_meta_updater import update_stock_metadata

        update_stock_metadata()
        logging.info("Stock metadata update completed successfully.")
    except Exception:
        logging.error("종목 메타데이터 갱신 작업이 실패했습니다.", exc_info=True)


def main():
    # 로깅 설정
    setup_logging()

    # Load .env for API keys, DB, etc.
    load_env_if_present()

    scheduler = BlockingScheduler()
    # 1. recommendation_cron 와 notify_cron 이 겹치는 경우: 작업도 실행되고, 슬랙도 발송
    # 2. python recommend.py 로 실행되는 경우: 작업도 실행되고, 슬랙도 발송
    # 3. recommendation_cron 에는 해당되지만 notify_cron 에 해당 안되는 경우: 작업은 실행되고, 슬랙은 발송안됨
    country_schedules = get_all_country_schedules()
    for schedule_name, cfg in country_schedules.items():
        if not cfg.get("enabled", True):
            logging.info("Skipping %s schedule (disabled)", schedule_name.upper())
            continue

        cron_expr_raw = cfg.get("signal_cron") or cfg.get("recommendation_cron")
        timezone = cfg.get("timezone") or TIMEZONE

        account_id = (cfg.get("account_id") or "").strip().lower()
        country_code = (cfg.get("country_code") or "").strip().lower()

        if not account_id or not country_code or not cron_expr_raw:
            raise RuntimeError(f"Schedule entry '{schedule_name}' must define account_id, country_code, and recommendation_cron")

        cron_expr = normalize_cron_weekdays(cron_expr_raw, target="apscheduler")
        if cron_expr != cron_expr_raw:
            logging.info(
                "Adjusted cron expression from '%s' to '%s' for APScheduler compatibility.",
                cron_expr_raw,
                cron_expr,
            )

        scheduler.add_job(
            run_recommendation_generation,
            CronTrigger.from_crontab(cron_expr, timezone=timezone),
            kwargs={
                "account_id": account_id,
                "country_code": country_code,
                "schedule_timezone": timezone,
            },
            id=f"{account_id}:{country_code}",
        )

    stats_cron = "0 1 * * *"  # 매일 새벽 1시에 종목 메타데이터 갱신
    scheduler.add_job(
        run_stock_stats_update,
        CronTrigger.from_crontab(stats_cron, timezone=TIMEZONE),
        id="stock_stats_update",
    )
    logging.info(f"Scheduled STATS UPDATE: cron='{stats_cron}' tz='{TIMEZONE}'")

    cache_cron = "0 2 * * *"  # 매일 새벽 2시에 가격 캐시 갱신
    scheduler.add_job(
        run_cache_refresh,
        CronTrigger.from_crontab(cache_cron, timezone=TIMEZONE),
        id="cache_refresh",
    )
    logging.info(f"Scheduled CACHE REFRESH: cron='{cache_cron}' tz='{TIMEZONE}'")

    # Initial run
    logging.info("\n[Initial Run] Starting...")

    # Initial run for stock metadata/cache refresh
    try:
        # 메타 데이터 갱신하고 싶을 때 해제
        run_stock_stats_update()
        # 서버 캐시 제거하고 싶을 때 해제
        run_cache_refresh()
    except Exception:
        logging.error("Error during initial run for stock metadata update/cache refresh", exc_info=True)

    for schedule_name, cfg in country_schedules.items():
        if not cfg.get("enabled", True):
            continue

        account_id = (cfg.get("account_id") or "").strip().lower()
        country_code = (cfg.get("country_code") or "").strip().lower()
        init_timezone = cfg.get("timezone") or TIMEZONE

        if not account_id or not country_code:
            raise RuntimeError(f"Schedule entry '{schedule_name}' must define both account_id and country_code")

        try:
            run_recommendation_generation(
                account_id,
                country_code=country_code,
                schedule_timezone=init_timezone,
            )
        except Exception:
            logging.error(
                "Error during initial run for account=%s country=%s",
                account_id,
                country_code,
                exc_info=True,
            )

    logging.info("[Initial Run] Complete.")

    # 다음 실행 시간 출력
    jobs = scheduler.get_jobs()
    if jobs:
        logging.info("\nNext scheduled run times:")
        for job in jobs:
            # 3.x: job.next_run_time
            next_time = getattr(job, "next_run_time", None)
            if next_time is not None:
                logging.info("- %s: %s", job.id, next_time.strftime("%Y-%m-%d %H:%M:%S"))

    logging.info("[APS] Scheduler started. Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
