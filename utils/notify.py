import os
from typing import Optional

import requests
import settings as global_settings

_LAST_ERROR: Optional[str] = None


def get_slack_webhook_url(country: str, account: Optional[str] = None) -> Optional[str]:
    """
    지정된 계정 또는 국가의 슬랙 웹훅 URL을 가져옵니다.
    1. 계정 설정(DB)의 SLACK_WEBHOOK_URL을 우선 확인합니다.
    2. 없으면, 국가별 환경 변수(예: KOR_SLACK_WEBHOOK)를 확인합니다.
    """
    from utils.db_manager import get_account_settings

    # 1. DB에서 계정별 설정 확인
    if account:
        settings = get_account_settings(account)
        if settings and settings.get("SLACK_WEBHOOK_URL"):
            return settings["SLACK_WEBHOOK_URL"]

    # 2. 환경 변수에서 국가별 설정 확인 (폴백)
    env_var_name = f"{country.upper()}_SLACK_WEBHOOK"
    return os.environ.get(env_var_name)


def send_log_to_slack(message: str):
    """중요 로그 메시지를 전용 슬랙 채널로 전송합니다."""
    webhook_url = os.environ.get("LOGS_SLACK_WEBHOOK")

    if webhook_url:
        log_message = f"📜 *[{global_settings.APP_TYPE}]* {message}"
        send_slack_message(log_message, webhook_url=webhook_url)


def send_verbose_log_to_slack(message: str):
    """잡다한 로그 메시지를 전용 슬랙 채널로 전송합니다."""
    webhook_url = os.environ.get("VERBOSE_LOGS_SLACK_WEBHOOK")

    if webhook_url:
        log_message = f"📜 *[{global_settings.APP_TYPE}]* {message}"
        send_slack_message(log_message, webhook_url=webhook_url)


def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def send_slack_message(
    text: str, blocks: Optional[list] = None, webhook_url: Optional[str] = None
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
        print(response.text)
        return response.text == "ok"
    except requests.exceptions.RequestException as e:
        _LAST_ERROR = str(e)
        return False
    except Exception as e:
        _LAST_ERROR = str(e)
        return False
