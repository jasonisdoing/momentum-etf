import os
from datetime import datetime
from typing import Optional, Tuple

import requests
import settings as global_settings

try:
    from croniter import croniter
    import pytz
except ImportError:
    croniter = None
    pytz = None

_LAST_ERROR: Optional[str] = None


def get_slack_webhook_url(country: str, account: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    지정된 계정 또는 국가의 슬랙 웹훅 URL과 이름을 가져옵니다.
    1. 계정 설정 파일의 SLACK_WEBHOOK_URL을 우선 확인합니다.
    2. 없으면, 국가별 환경 변수(예: KOR_SLACK_WEBHOOK)를 확인합니다.
    """
    from utils.account_registry import get_account_file_settings

    # 1. 파일에서 계정별 설정 확인
    if account:
        try:
            settings = get_account_file_settings(account)
            url = settings.get("slack_webhook_url")
            if url:
                # 파일에서 가져온 경우, 이름은 계정명으로 합니다.
                return url, f"account:{account}"
        except SystemExit:
            pass

    # 2. 환경 변수에서 국가별 설정 확인 (폴백)
    env_var_name = f"{country.upper()}_SLACK_WEBHOOK"
    url = os.environ.get(env_var_name)
    if url:
        return url, env_var_name

    return None


def should_notify_on_schedule(country: str) -> bool:
    """
    알림 전용 CRON 설정이 있는 경우, 현재 시간이 해당 스케줄과 일치하는지 확인합니다.
    설정이 없으면 항상 True를 반환합니다.
    """
    if not croniter or not pytz:
        return True  # 라이브러리가 없으면 검사를 건너뛰고 항상 알림

    # 1. 알림 전용 CRON 환경 변수 확인
    # 환경 변수를 먼저 확인하고, 없으면 settings.py의 설정을 사용합니다.
    notify_cron_env = f"NOTIFY_{country.upper()}_CRON"
    cron_schedule = os.environ.get(notify_cron_env)
    if cron_schedule is None:
        # settings.py에서 해당 변수를 가져옵니다.
        cron_schedule = getattr(global_settings, notify_cron_env, None)

    if not cron_schedule:
        return True  # 설정이 없으면 항상 알림

    # 2. 타임존 설정 (aps.py와 일관성 유지)
    tz_env = f"SCHEDULE_{country.upper()}_TZ"
    default_tz = {"kor": "Asia/Seoul", "aus": "Australia/Sydney", "coin": "Asia/Seoul"}.get(
        country, "Asia/Seoul"
    )
    tz_str = os.environ.get(tz_env, default_tz)

    try:
        local_tz = pytz.timezone(tz_str)
        now = datetime.now(local_tz)
        return croniter.match(cron_schedule, now)
    except Exception:
        return True  # 오류 발생 시 안전하게 알림 허용


def send_log_to_slack(message: str):
    """중요 로그 메시지를 전용 슬랙 채널로 전송합니다."""
    webhook_url = os.environ.get("LOGS_SLACK_WEBHOOK")

    if webhook_url:
        log_message = f"📜 *[{global_settings.APP_TYPE}]* {message}"
        send_slack_message(log_message, webhook_url=webhook_url, webhook_name="LOGS_SLACK_WEBHOOK")


def send_verbose_log_to_slack(message: str):
    """잡다한 로그 메시지를 전용 슬랙 채널로 전송합니다."""
    webhook_url = os.environ.get("VERBOSE_LOGS_SLACK_WEBHOOK")

    if webhook_url:
        log_message = f"📜 *[{global_settings.APP_TYPE}]* {message}"
        send_slack_message(
            log_message, webhook_url=webhook_url, webhook_name="VERBOSE_LOGS_SLACK_WEBHOOK"
        )


def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def send_slack_message(
    text: str,
    blocks: Optional[list] = None,
    webhook_url: Optional[str] = None,
    webhook_name: Optional[str] = None,
) -> bool:
    """Send a Slack message using the provided webhook URL."""
    global _LAST_ERROR
    _LAST_ERROR = None
    if not webhook_url:
        _LAST_ERROR = "Missing Slack webhook URL"
        return False

    if blocks:
        payload = {"blocks": blocks}
    else:
        # For plain text, convert HTML <pre> to slack's ```
        text = text.replace("<pre>", "```").replace("</pre>", "```")
        payload = {"text": text}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        if response.text == "ok":
            print(f"[SLACK] 메시지 전송 - {webhook_name or 'Unknown Webhook'}")
            return True
        return False
    except requests.exceptions.RequestException as e:
        _LAST_ERROR = str(e)
        return False
    except Exception as e:
        _LAST_ERROR = str(e)
        return False
