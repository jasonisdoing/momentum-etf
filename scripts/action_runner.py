#!/usr/bin/env python
"""
GitHub Actions 전용 실행 스크립트.
주어진 국가 시장의 휴장일 여부를 파악하고, 영업일인 경우에만 추천 로직을 실행.
휴장일인 경우 Slack으로 휴식 메시지 전송.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pytz

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommend import run_recommendation_generation_v2 as run_recommendation_generation
from utils.account_registry import get_account_settings, list_available_accounts
from utils.data_loader import get_trading_days
from utils.env import load_env_if_present
from utils.notification import send_recommendation_slack_notification


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_today_str_for_country(country: str) -> str:
    from config import MARKET_SCHEDULES

    schedule = MARKET_SCHEDULES.get(country)
    if schedule and schedule.get("timezone"):
        tz = pytz.timezone(schedule["timezone"])
        return datetime.now(tz).strftime("%Y-%m-%d")
    # 기본은 KST
    return datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d")


def main():
    setup_logger()
    load_env_if_present()

    parser = argparse.ArgumentParser(description="Action Runner for momentum-etf")
    parser.add_argument("--country", required=True, help="Country code (e.g., kor, us, au)")
    args = parser.parse_args()

    country = str(args.country).strip().lower()

    # 1. 대상 국가의 로컬 타임존 기준으로 오늘 날짜 구하기
    today_str = get_today_str_for_country(country)

    # 2. 해당 국가를 타겟으로 하는 계정 모으기
    target_accounts = []
    for acc in list_available_accounts():
        settings = get_account_settings(acc)
        if settings.get("country_code", "kor").strip().lower() == country:
            target_accounts.append(acc)

    if not target_accounts:
        logging.warning(f"실행할 계정이 없습니다. (country: {country})")
        return

    # 3. 휴장일 여부 확인 (최근 거래일 목록에서 오늘이 포함되는지)
    try:
        trading_days = get_trading_days(today_str, today_str, country)
        is_trading_day = len(trading_days) > 0
    except Exception as e:
        logging.error(f"휴장일 조회 실패: {e}. 안전을 위해 영업일로 간주하고 진행합니다.")
        is_trading_day = True

    if not is_trading_day:
        logging.info(f"[{country.upper()}] 오늘은 휴장일입니다. ({today_str})")
        for acc in target_accounts:
            try:
                msg = f"🏖️ 오늘은 {country.upper()} 시장 휴장일입니다.\n포트폴리오 점검은 내려놓고 푹 쉬세요!"
                send_recommendation_slack_notification(msg)
                logging.info(f"[{acc}] 휴장일 알림 전송 완료")
            except Exception as e:
                logging.error(f"[{acc}] 휴장일 알림 전송 실패: {e}")
        return

    # 4. 영업일이면 추천 로직 실행
    logging.info(f"[{country.upper()}] 영업일입니다. 추천 로직을 시작합니다.")
    for acc in target_accounts:
        try:
            run_recommendation_generation(acc, send_slack=True)
        except Exception as e:
            logging.error(f"[{acc}] 추천 로직 실행 중 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()
