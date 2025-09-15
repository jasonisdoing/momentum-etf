from typing import Optional

import requests

_LAST_ERROR: Optional[str] = None


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
        return response.text == "ok"
    except requests.exceptions.RequestException as e:
        _LAST_ERROR = str(e)
        return False
    except Exception as e:
        _LAST_ERROR = str(e)
        return False
