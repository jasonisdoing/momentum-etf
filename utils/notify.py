import json
import urllib.request
import urllib.error
from typing import Optional

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
