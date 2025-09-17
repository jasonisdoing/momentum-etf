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
from datetime import datetime

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


def run_status(country: str):
    """Run status generation and implicit Slack notification."""
    try:
        from status import main as run_status_main

        print(f"Running status for {country}")
        if country == "coin":
            _try_sync_bithumb_trades()
            _try_sync_bithumb_equity()

        run_status_main(country=country, date_str=None)
    except Exception as e:
        print(f"Status job failed for {country}: {e}")


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
        print(f"snapshot_main----------------")
        snapshot_main()
        print(f"snapshot_main----------------")

        # 3. 업데이트 후 새로운 평가금액을 가져옵니다.
        new_snapshot = get_portfolio_snapshot("coin")
        new_equity = float(new_snapshot.get("total_equity", 0.0)) if new_snapshot else 0.0

        # 4. 스케줄러에 의한 업데이트임을 기록하기 위해 `updated_by`와 함께 항상 저장합니다.
        if new_snapshot:
            save_daily_equity("coin", new_snapshot["date"], new_equity, updated_by="스케줄러")
            print(f"-> 코인 평가금액 스냅샷 업데이트 완료. (updated_by='스케줄러')")

            # 5. 평가금액이 변경되었는지 확인하고, 변경된 경우 슬랙 알림을 보냅니다.
            if abs(new_equity - old_equity) > 1e-9:
                print(
                    f"-> 코인 평가금액 변경 감지: {old_equity:,.0f}원 -> {new_equity:,.0f}원. 알림을 보냅니다."
                )
                _notify_equity_update("coin", old_equity, new_equity)
            else:
                print("-> 코인 평가금액에 변경이 없습니다.")
        else:
            print("-> 코인 평가금액 스냅샷을 찾을 수 없어 업데이트를 건너뜁니다.")

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
        scheduler.add_job(
            run_status,
            CronTrigger.from_crontab(cron, timezone=tz),
            args=["coin"],
            id="coin",
        )
        print(f"Scheduled COIN: cron={cron} tz={tz}")

    # aus
    if _bool_env("SCHEDULE_ENABLE_AUS", True):
        cron = common.get("SCHEDULE_CRON_AUS") or _get("SCHEDULE_AUS_CRON", "10 18 * * 1-5")
        tz = _get("SCHEDULE_KOR_TZ", "Asia/Seoul")
        scheduler.add_job(
            run_status,
            CronTrigger.from_crontab(cron, timezone=tz),
            args=["aus"],
            id="aus",
        )
        print(f"Scheduled AUS: cron={cron} tz={tz}")

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
        print(f"Scheduled KOR: cron={cron} tz={tz}")

    if _bool_env("RUN_IMMEDIATELY_ON_START", False):
        # 시작 시 한 번 즉시 실행
        print("\n[초기 실행] 시작...")
        for c in ("coin", "aus", "kor"):
            try:
                if _bool_env(f"SCHEDULE_ENABLE_{c.upper()}", True):
                    run_status(c)
            except Exception as e:
                print(f"초기 실행 중 오류 ({c}): {e}")
        print("[초기 실행] 완료.")

    # 다음 실행 시간 출력
    jobs = scheduler.get_jobs()
    if jobs:
        print("\n다음 실행 예정 시간:")
        for job in jobs:
            # 3.x: job.next_run_time
            next_time = getattr(job, "next_run_time", None)

            # 4.x (혹시 모르게 혼합된 경우): trigger에서 직접 구하기
            if next_time is None and hasattr(job, "trigger"):
                next_time = job.trigger.get_next_fire_time(None, datetime.now())

            if next_time:
                print(f"- {job.id}: {_format_korean_datetime(next_time)}")
            else:
                print(f"- {job.id}: 실행 예정 없음")
    print("\n스케줄러를 시작합니다. 다음 주기까지 대기합니다...")
    scheduler.start()


if __name__ == "__main__":
    main()
