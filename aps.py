"""
APScheduler 기반 자동화 작업 모음

작업 요약 (모두 KST 기준):
- K1(한국 ETF) 추천: 월~금 09:01~16:51, 10분 간격으로 실행
- 가격 캐시 갱신: 매일 04:00 실행
- 프로세스 기동 시 모든 계정 추천 1회 즉시 실행(설정 허용 시)

스케줄 정의는 zsettings/account/<account>.json 의 schedule 섹션을 참고합니다.
"""

# 프로젝트 루트를 Python 경로에 추가
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

TIMEZONE = "Asia/Seoul"

# pkg_resources 워닝 억제
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import time
from datetime import datetime, time as dt_time

from utils.recommendation_storage import save_recommendation_report

from utils.market_schedule import generate_market_cron_expressions


try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

try:  # pragma: no cover - 선택적 의존성 처리
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from logic.recommend.pipeline import RecommendationReport, generate_account_recommendation_report
from utils.cron_utils import normalize_cron_weekdays
from utils.env import load_env_if_present
from utils.notification import compose_recommendation_slack_message, send_recommendation_slack_notification
from utils.schedule_config import get_all_country_schedules, get_global_schedule_settings


def setup_logging() -> None:
    """
    로그 파일을 설정합니다. logs/YYYY-MM-DD.log 형식으로 생성됩니다.
    프로세스가 시작될 때의 날짜를 기준으로 파일명이 정해집니다.
    """
    # 프로젝트 루트 아래에 logs 디렉토리 생성
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # YYYY-MM-DD.log 파일명 설정
    now_kst = _now_kst()

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
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def run_recommendation_generation(account_id: str, *, country_code: str) -> RecommendationReport:
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
        message = compose_recommendation_slack_message(
            account_id,
            report,
            duration=elapsed,
        )
        notified = send_recommendation_slack_notification(account_id, message)
        if notified:
            logging.info(
                "[%s/%s] Slack 알림 전송이 완료되었습니다 (소요 %.2fs)",
                country_code.upper(),
                getattr(report, "base_date", "N/A"),
                elapsed,
            )
        else:
            logging.info(
                "[%s/%s] Slack 알림이 전송되지 않았습니다.",
                country_code.upper(),
                getattr(report, "base_date", "N/A"),
            )
    except Exception:
        logging.error("Slack 알림 전송 실패", exc_info=True)

    return report


def run_cache_refresh() -> None:
    """모든 국가의 가격 캐시를 갱신합니다."""
    from utils.account_registry import get_common_file_settings

    common_settings = get_common_file_settings()
    start_date = str(common_settings.get("CACHE_START_DATE"))
    countries = ["kor"]
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


@dataclass(frozen=True)
class RecommendationJobConfig:
    schedule_name: str
    account_id: str
    country_code: str
    cron_exprs: Tuple[str, ...]
    timezone: str
    run_immediately: bool


def _now_kst() -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(TIMEZONE))
        except Exception:
            pass
    return datetime.now()


def _validate_timezone(tz_value: Optional[str]) -> str:
    tz = (tz_value or "").strip()
    if not tz:
        return TIMEZONE
    if tz != TIMEZONE:
        logging.warning("Timezone '%s'은 지원하지 않아 '%s'로 대체합니다.", tz, TIMEZONE)
        return TIMEZONE
    return tz


def _load_recommendation_jobs() -> Tuple[RecommendationJobConfig, ...]:
    schedules = get_all_country_schedules()
    global_settings = get_global_schedule_settings()
    run_immediately_default = bool(global_settings.get("run_immediately_on_start", False))

    jobs: list[RecommendationJobConfig] = []
    for schedule_name, cfg in schedules.items():
        if not cfg.get("enabled", True):
            logging.info("Skipping %s schedule (disabled)", schedule_name.upper())
            continue

        account_id = (cfg.get("account_id") or "").strip().lower()
        country_code = (cfg.get("country_code") or "").strip().lower()
        if not account_id or not country_code:
            logging.warning("Schedule '%s'에 account_id/country_code가 없어 건너뜁니다.", schedule_name)
            continue

        # config.py의 MARKET_SCHEDULES에서 크론 표현식 자동 생성
        cron_list = tuple(cfg.get("recommendation_cron_list") or [])
        if not cron_list:
            try:
                cron_list = tuple(generate_market_cron_expressions(country_code))
            except ValueError as exc:
                logging.warning("Schedule '%s': %s", schedule_name, exc)
                continue

        timezone = _validate_timezone(cfg.get("timezone"))
        run_immediately = bool(cfg.get("run_immediately_on_start", run_immediately_default))

        jobs.append(
            RecommendationJobConfig(
                schedule_name=schedule_name,
                account_id=account_id,
                country_code=country_code,
                cron_exprs=cron_list,
                timezone=timezone,
                run_immediately=run_immediately,
            )
        )

    return tuple(jobs)


def _register_recommendation_jobs(scheduler: BlockingScheduler, jobs: Iterable[RecommendationJobConfig]) -> None:
    for job in jobs:
        for index, cron_expr in enumerate(job.cron_exprs, start=1):
            scheduler.add_job(
                run_recommendation_generation,
                CronTrigger.from_crontab(cron_expr, timezone=job.timezone),
                kwargs={"account_id": job.account_id, "country_code": job.country_code},
                id=f"{job.account_id}:{job.country_code}:{index}",
            )

        if job.cron_exprs:
            if len(job.cron_exprs) == 1:
                cron_summary = job.cron_exprs[0]
            else:
                cron_summary = f"{job.cron_exprs[0]} … {job.cron_exprs[-1]}"
            logging.info(
                "Scheduled RECOMMENDATION: schedule=%s account=%s country=%s cron='%s' (total %d slots) tz='%s'",
                job.schedule_name.upper(),
                job.account_id.upper(),
                job.country_code.upper(),
                cron_summary,
                len(job.cron_exprs),
                job.timezone,
            )


def _register_cache_job(
    scheduler: BlockingScheduler,
    *,
    hourly_cron_expr: str = "0 * * * *",
) -> None:
    scheduler.add_job(
        run_cache_refresh,
        CronTrigger.from_crontab(hourly_cron_expr, timezone=TIMEZONE),
        id="cache_refresh_hourly",
    )
    logging.info("Scheduled CACHE REFRESH (hourly): cron='%s' tz='%s'", hourly_cron_expr, TIMEZONE)


def _run_initial_recommendations(jobs: Iterable[RecommendationJobConfig]) -> None:
    executable = [job for job in jobs if job.run_immediately]
    if not executable:
        logging.info("[Initial Run] Executing %d recommendation job(s)...", len(jobs))
        executable = list(jobs)
    else:
        logging.info("[Initial Run] Executing %d recommendation job(s)...", len(executable))
    for job in executable:
        try:
            run_recommendation_generation(job.account_id, country_code=job.country_code)
        except Exception:
            logging.error(
                "Error during initial run for account=%s country=%s",
                job.account_id,
                job.country_code,
                exc_info=True,
            )
    logging.info("[Initial Run] Complete.")


def _log_next_runs(scheduler: BlockingScheduler) -> None:
    jobs = scheduler.get_jobs()
    if not jobs:
        logging.info("No jobs registered.")
        return
    now = datetime.now(ZoneInfo(TIMEZONE)) if ZoneInfo else datetime.now()

    logging.info("Next scheduled run times:")
    for job in jobs:
        try:
            next_time = job.next_run_time
        except AttributeError:
            next_time = None

        if next_time is None:
            try:
                next_time = job.trigger.get_next_fire_time(None, now)
            except Exception:
                next_time = None

        if next_time is not None:
            logging.info("- %s: %s", job.id, next_time.strftime("%Y-%m-%d %H:%M:%S"))


def main() -> None:
    # 로깅 설정
    setup_logging()

    # Load .env for API keys, DB, etc.
    load_env_if_present()

    scheduler = BlockingScheduler()

    jobs = _load_recommendation_jobs()
    if not jobs:
        logging.warning("등록 가능한 추천 잡이 없습니다. 설정을 확인하세요.")

    logging.info("Running initial price cache refresh before scheduling recommendations...")
    run_cache_refresh()

    _register_recommendation_jobs(scheduler, jobs)
    _register_cache_job(scheduler)

    _run_initial_recommendations(jobs)
    _log_next_runs(scheduler)

    logging.info("[APS] Scheduler started. Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
