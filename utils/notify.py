import io
import json
import urllib.error
import urllib.request
from typing import Optional

import requests

from utils.db_manager import get_common_settings

_LAST_ERROR: Optional[str] = None


def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def _get_telegram_config():
    cfg = get_common_settings() or {}
    enabled = bool(cfg.get("TELEGRAM_ENABLED", False))
    token = cfg.get("TELEGRAM_BOT_TOKEN")
    chat_id = cfg.get("TELEGRAM_CHAT_ID")
    return enabled, token, chat_id


def _get_slack_config():
    cfg = get_common_settings() or {}
    enabled = bool(cfg.get("SLACK_ENABLED", False))
    webhook_url = cfg.get("SLACK_WEBHOOK_URL")
    return enabled, webhook_url


def send_telegram_message(text: str) -> bool:
    """Send a Telegram message if enabled and properly configured.

    Returns True on success, False otherwise. Uses urllib to avoid extra deps.
    """
    global _LAST_ERROR
    _LAST_ERROR = None
    enabled, token, chat_id = _get_telegram_config()
    if not enabled:
        _LAST_ERROR = "TELEGRAM_ENABLED is False"
        return False
    if not token or not chat_id:
        # Misconfigured; don't attempt network.
        _LAST_ERROR = "Missing token or chat_id"
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": str(chat_id),
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
            if not ok:
                try:
                    body = resp.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = ""
                _LAST_ERROR = f"HTTP {resp.status} {body}"
            return ok
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        _LAST_ERROR = f"HTTPError {e.code}: {body}".strip()
        return False
    except Exception as e:
        _LAST_ERROR = str(e)
        return False


def send_telegram_photo(caption: str, image_buffer: io.BytesIO) -> bool:
    """Send a photo with a caption to Telegram."""
    global _LAST_ERROR
    _LAST_ERROR = None
    enabled, token, chat_id = _get_telegram_config()
    if not enabled:
        _LAST_ERROR = "TELEGRAM_ENABLED is False"
        return False
    if not token or not chat_id:
        _LAST_ERROR = "Missing token or chat_id"
        return False

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    files = {"photo": image_buffer.getvalue()}
    data = {"chat_id": str(chat_id), "caption": caption, "parse_mode": "HTML"}

    try:
        response = requests.post(url, files=files, data=data, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json().get("ok", False)
    except requests.exceptions.RequestException as e:
        _LAST_ERROR = str(e)
        return False
    except Exception as e:
        _LAST_ERROR = str(e)
        return False


def send_slack_message(text: str, blocks: Optional[list] = None) -> bool:
    """Send a Slack message if enabled and properly configured."""
    global _LAST_ERROR
    _LAST_ERROR = None
    enabled, webhook_url = _get_slack_config()
    if not enabled:
        _LAST_ERROR = "SLACK_ENABLED is False"
        return False
    if not webhook_url:
        _LAST_ERROR = "Missing Slack webhook URL"
        return False

    if blocks:
        payload = {"blocks": blocks}
    else:
        # For plain text, convert telegram's <pre> to slack's ```
        text = text.replace("<pre>", "```").replace("</pre>", "```")
        payload = {"text": text}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return response.text == "ok"
    except requests.exceptions.RequestException as e:
        _LAST_ERROR = str(e)
        return False
    except Exception as e:
        _LAST_ERROR = str(e)
        return False
