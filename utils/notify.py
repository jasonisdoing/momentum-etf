import os
from typing import Optional, Tuple

import requests
import settings as global_settings

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
            settings = get_account_file_settings(country, account)
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
