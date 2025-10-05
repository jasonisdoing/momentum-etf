"""
APScheduler 기반 스케줄러

[스케줄 설정]
스케줄은 아래 환경 변수를 통해 설정할 수 있습니다.
환경 변수가 없으면 각 작업의 기본값(Default)이 사용됩니다.

- SCHEDULE_ENABLE_KOR/AUS: "1" 또는 "0" (기본: "1", 활성화)
- SCHEDULE_KOR_CRON: 한국 추천 계산 주기
- SCHEDULE_AUS_CRON: 호주 추천 계산 주기\
- SCHEDULE_KOR_TZ: 한국 시간대 (기본: "Asia/Seoul")
- SCHEDULE_AUS_TZ: 호주 시간대 (기본: "Asia/Seoul")\
- RUN_IMMEDIATELY_ON_START: "1" 이면 시작 시 즉시 한 번 실행 (기본: "0")
"""

import json
import logging
import os
import sys
import warnings

# pkg_resources 워닝 억제
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import time
from datetime import datetime
from pathlib import Path


try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
except Exception:  # pragma: no cover - 선택적 의존성 처리
    WebClient = None  # type: ignore[assignment]
    SlackApiError = Exception  # type: ignore[assignment]

import pandas as pd

try:  # pragma: no cover - 선택적 의존성 처리
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from logic.recommend.pipeline import (
    RecommendationReport,
    generate_account_recommendation_report,
)
from utils.data_updater import update_etf_names
from utils.env import load_env_if_present
from utils.notification import (
    compose_recommendation_slack_message,
    get_slack_webhook_url,
    send_slack_message,
)
from utils.schedule_config import (
    get_all_country_schedules,
    get_cache_schedule,
    get_global_schedule_settings,
)
from utils.settings_loader import get_account_slack_channel


RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _make_json_safe(obj):
    """Convert non-serializable objects into JSON safe representations."""

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()

    if np is not None and isinstance(obj, np.generic):  # numpy scalar types
        return obj.item()

    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]

    if isinstance(obj, pd.Series):
        return [_make_json_safe(v) for v in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [
            {k: _make_json_safe(v) for k, v in rec.items()} for rec in obj.to_dict(orient="records")
        ]

    return str(obj)


def _save_recommendation_result(report: RecommendationReport) -> Path:
    """Persist recommendation results to JSON for downstream consumers."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    account_id = (getattr(report, "account_id", "") or "").strip().lower()
    country_code = (getattr(report, "country_code", "") or "").strip().lower()

    if not account_id or not country_code:
        raise RuntimeError("Recommendation report must include both account_id and country_code")

    payload = _make_json_safe(report.recommendations)

    account_path = RESULTS_DIR / f"recommendation_{account_id}.json"
    country_path = RESULTS_DIR / f"recommendation_{country_code}.json"

    try:
        with account_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        if country_path != account_path:
            with country_path.open("w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
    except Exception:
        logging.error(
            "Failed to write recommendation results for account=%s country=%s",
            account_id,
            country_code,
            exc_info=True,
        )
        raise

    return account_path


def _send_slack_notification(
    account_id: str,
    country_code: str,
    message: str,
) -> bool:
    """Send Slack notification via bot token or webhook as a fallback."""

    channel = get_account_slack_channel(account_id)
    token = os.environ.get("SLACK_BOT_TOKEN")

    if token and channel and WebClient is not None:
        try:
            client = WebClient(token=token)
            client.chat_postMessage(channel=channel, text=message)
            logging.info(
                "Slack message sent via bot token for account=%s (channel=%s)",
                account_id,
                channel,
            )
            return True
        except SlackApiError as exc:  # pragma: no cover - 외부 API 호출 오류
            logging.error(
                "Slack API error for account=%s: %s",
                account_id,
                getattr(exc, "response", {}).get("error") or str(exc),
                exc_info=True,
            )
        except Exception:
            logging.error(
                "Unexpected Slack client failure for account=%s",
                account_id,
                exc_info=True,
            )

    webhook_info = get_slack_webhook_url(account_id)
    if webhook_info:
        webhook_url, source_name = webhook_info
        sent = send_slack_message(message, webhook_url=webhook_url, webhook_name=source_name)
        if sent:
            logging.info(
                "Slack message sent via webhook for %s (source=%s)",
                country_code.upper(),
                source_name,
            )
            return True

        logging.error(
            "Slack webhook delivery failed for account=%s (source=%s)",
            account_id,
            source_name,
        )

    if not channel and not webhook_info:
        logging.info(
            "Slack delivery skipped for account=%s: no channel/webhook configured",
            account_id,
        )

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


def run_recommendation_generation(
    account_id: str,
    *,
    country_code: str,
    force_notify: bool = False,
) -> None:
    """Generate recommendations, persist them, and notify Slack."""

    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        logging.error("Account ID is required for recommendation generation.")
        return

    country_norm = (country_code or "").strip().lower()
    if not country_norm:
        logging.error("Country code is required for recommendation generation.")
        return

    logging.info(
        "Running recommendation generation for account=%s (country=%s)",
        account_norm,
        country_norm,
    )
    start_time = time.time()

    try:
        report = generate_account_recommendation_report(account_id=account_norm, date_str=None)
    except Exception:
        logging.error(
            "Signal generation job for account=%s failed",
            account_norm,
            exc_info=True,
        )
        return

    if not isinstance(report, RecommendationReport):
        logging.error(
            "Unexpected recommendation report type for account=%s: %s",
            account_norm,
            type(report).__name__,
        )
        return

    if not report.recommendations:
        logging.warning("No recommendations produced for account=%s", account_norm)
        return

    duration = time.time() - start_time

    report_country = (getattr(report, "country_code", "") or "").strip().lower()
    if report_country and report_country != country_norm:
        logging.warning(
            "Report country mismatch (expected=%s, got=%s)",
            country_norm,
            report_country,
        )

    target_country = report_country or country_norm

    try:
        output_path = _save_recommendation_result(report)
        logging.info(
            "Saved %s recommendations (%d items) to %s",
            target_country.upper(),
            len(report.recommendations),
            output_path,
        )
    except Exception:
        logging.error(
            "Skipping Slack notification because saving results failed for account=%s",
            account_norm,
            exc_info=True,
        )
        return

    slack_message = compose_recommendation_slack_message(
        account_norm,
        report,
        duration=duration,
        force_notify=force_notify,
    )

    notified = _send_slack_notification(
        account_norm,
        target_country,
        slack_message,
    )
    base_date_str = report.base_date.strftime("%Y-%m-%d")
    if notified:
        logging.info(
            "[%s/%s] Slack notification completed in %.1fs",
            target_country.upper(),
            base_date_str,
            duration,
        )
    else:
        logging.info(
            "[%s/%s] Slack notification skipped or failed",
            target_country.upper(),
            base_date_str,
        )


def run_recommend_for_country(
    account_id: str,
    country: str,
    *,
    force_notify: bool = False,
) -> None:
    account_norm = (account_id or "").strip().lower()
    country_norm = (country or "").strip().lower()

    if not account_norm:
        logging.error("Account ID must be provided when scheduling recommendations.")
        return

    try:
        run_recommendation_generation(
            account_norm,
            country_code=country_norm,
            force_notify=force_notify,
        )
    except Exception:
        logging.error(
            "Error running recommendation generation for account=%s country=%s",
            account_norm,
            country_norm,
            exc_info=True,
        )


def run_cache_refresh() -> None:
    """모든 국가의 가격 캐시를 갱신합니다."""
    start_date = os.environ.get("CACHE_START_DATE", "2020-01-01")
    countries_env = os.environ.get("CACHE_COUNTRIES", "kor,aus")
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
    # 1. recommendation_cron 와 notify_cron 이 겹치는 경우: 작업도 실행되고, 슬랙도 발송
    # 2. python recommend.py 로 실행되는 경우: 작업도 실행되고, 슬랙도 발송
    # 3. recommendation_cron 에는 해당되지만 notify_cron 에 해당 안되는 경우: 작업은 실행되고, 슬랙은 발송안됨
    cron_default = "0 0 * * *"
    tz_default = "Asia/Seoul"
    country_schedules = get_all_country_schedules()
    for schedule_name, cfg in country_schedules.items():
        enabled_default = bool(cfg.get("enabled", True))
        if not _bool_env(f"SCHEDULE_ENABLE_{schedule_name.upper()}", enabled_default):
            logging.info(f"Skipping {schedule_name.upper()} schedule (disabled)")
            continue

        cron_expr = _get(
            f"SCHEDULE_{schedule_name.upper()}_CRON",
            cfg.get("signal_cron") or cfg.get("recommendation_cron") or cron_default,
        )
        timezone = _get(
            f"SCHEDULE_{schedule_name.upper()}_TZ",
            cfg.get("timezone", tz_default),
        )

        account_id = (cfg.get("account_id") or "").strip().lower()
        country_code = (cfg.get("country_code") or "").strip().lower()

        if not account_id or not country_code:
            raise RuntimeError(
                f"Schedule entry '{schedule_name}' must define both account_id and country_code"
            )

        scheduler.add_job(
            run_recommend_for_country,
            CronTrigger.from_crontab(cron_expr, timezone=timezone),
            args=[account_id, country_code],
            id=f"{account_id}:{country_code}",
        )
        logging.info(
            "Scheduled %s (account=%s): cron='%s' tz='%s'",
            country_code.upper(),
            account_id,
            cron_expr,
            timezone,
        )

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
        for schedule_name, cfg in country_schedules.items():
            enabled_default = bool(cfg.get("enabled", True))
            if not _bool_env(f"SCHEDULE_ENABLE_{schedule_name.upper()}", enabled_default):
                continue

            account_id = (cfg.get("account_id") or "").strip().lower()
            country_code = (cfg.get("country_code") or "").strip().lower()

            if not account_id or not country_code:
                raise RuntimeError(
                    f"Schedule entry '{schedule_name}' must define both account_id and country_code"
                )

            try:
                run_recommend_for_country(account_id, country_code, force_notify=True)
            except Exception:
                logging.error(
                    "Error during initial run for account=%s country=%s",
                    account_id,
                    country_code,
                    exc_info=True,
                )
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
