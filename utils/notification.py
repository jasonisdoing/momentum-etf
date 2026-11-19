"""Slack notification helpers consolidated in one module."""

from __future__ import annotations

import os
import re
from typing import Any

import requests
from requests.adapters import HTTPAdapter

try:
    from slack_sdk import WebClient  # type: ignore
    from slack_sdk.errors import SlackApiError  # type: ignore
except Exception:  # pragma: no cover - 선택적 의존성 처리
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore

try:
    import pytz
    from croniter import croniter
except ImportError:  # pragma: no cover - 선택적 의존성 처리
    croniter = None
    pytz = None

from utils.logger import get_app_logger
from utils.report import format_kr_money
from utils.settings_loader import get_slack_channel

_LAST_ERROR: str | None = None
logger = get_app_logger()

# 슬랙 API 호출마다 TCP/TLS 핸드셰이크를 반복하지 않도록 전역 커넥션 풀을 재사용한다.
_SLACK_HTTP_SESSION = requests.Session()
_SLACK_HTTP_SESSION.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=20))

_SLACK_METHOD_PATHS = {
    "chat_postMessage": "chat.postMessage",
    "chat_update": "chat.update",
    "chat_delete": "chat.delete",
    "views_open": "views.open",
    "views_update": "views.update",
    "reactions_add": "reactions.add",
    "reactions_remove": "reactions.remove",
}


class PooledSlackClient:
    """전역 `requests.Session` 커넥션 풀을 공유하는 Slack Web API 경량 클라이언트."""

    def __init__(self, token: str) -> None:
        self._token = token

    def _call(self, api_method: str, **kwargs: Any) -> dict[str, Any]:
        payload = {key: value for key, value in kwargs.items() if value is not None}
        response = _SLACK_HTTP_SESSION.post(
            f"https://slack.com/api/{api_method}",
            headers={"Authorization": f"Bearer {self._token}"},
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise SlackApiError(f"Slack API error ({api_method}): {data.get('error')}", data)
        return data

    def __getattr__(self, name: str):
        api_method = _SLACK_METHOD_PATHS.get(name)
        if api_method is None:
            raise AttributeError(f"지원하지 않는 슬랙 API 메서드입니다: {name}")
        return lambda **kwargs: self._call(api_method, **kwargs)


def get_slack_client(token: str) -> PooledSlackClient:
    """커넥션 풀을 재사용하는 슬랙 클라이언트를 반환한다."""
    return PooledSlackClient(token)


# 배치가 스스로 슬랙 알림을 보냈을 때 쓰는 종료 코드.
# `infra/cron/run_batch.py` 래퍼가 이 코드를 보면 로그 꼬리를 붙인 일반 실패 알림을 보내지 않는다.
# (래퍼는 프로젝트 임포트 없이 도는 독립 스크립트라 같은 값을 자체 상수로 들고 있다.)
EXIT_ALREADY_NOTIFIED = 66


# ---------------------------------------------------------------------------
# 슬랙 웹훅 관련 헬퍼
# ---------------------------------------------------------------------------


def send_slack_message(
    text: str,
    blocks: list[dict] | None = None,
    webhook_url: str | None = None,
    webhook_name: str | None = None,
) -> bool:
    """Send a Slack message using the provided webhook URL."""

    global _LAST_ERROR
    _LAST_ERROR = None

    if not webhook_url:
        _LAST_ERROR = "Missing Slack webhook URL"
        return False

    sanitized = text.replace("<pre>", "```").replace("</pre>", "```")
    if blocks:
        payload = {"text": sanitized, "blocks": blocks}
    else:
        payload = {"text": sanitized}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        if response.text == "ok":
            logger.info("[SLACK] 메시지 전송 - %s", webhook_name or "Unknown Webhook")
            return True
        _LAST_ERROR = response.text or "unknown"
        return False
    except requests.exceptions.RequestException as exc:  # pragma: no cover - 단순 위임 예외 처리
        _LAST_ERROR = str(exc)
        return False
    except Exception as exc:  # pragma: no cover - 방어적 처리
        _LAST_ERROR = str(exc)
        return False


def get_last_error() -> str | None:
    return _LAST_ERROR


# ---------------------------------------------------------------------------
# 포맷팅 보조 함수
# ---------------------------------------------------------------------------


def strip_html_tags(value: str) -> str:
    try:
        return re.sub(r"<[^>]+>", "", value)
    except Exception:  # pragma: no cover - 방어적 처리
        return value


def app_link(path: str, label: str) -> str:
    """슬랙 메시지의 화면 링크 — 주소는 APP_BASE_URL 하나로만 정한다.

    도메인을 코드에 적으면 옮길 때마다 스크립트를 고쳐야 한다. 값이 없으면 링크
    없이 라벨만 남긴다(임의 주소를 만들지 않는다).
    """
    base = str(os.environ.get("APP_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return label
    return f"<{base}/{path.lstrip('/')}|{label}>"


def _send_slack_message_via_bot(
    payload: dict[str, Any] | str,
    thread_ts: str | None = None,
    *,
    channel: str | None,
) -> str | None:
    """
    전달받은 페이로드를 슬랙으로 전송합니다.
    성공 시 메시지의 ts(timestamp)를 반환하고, 실패 시 None을 반환합니다.
    """

    if isinstance(payload, str):
        text = payload
        blocks: list[dict[str, Any]] | None = None
    else:
        text = str(payload.get("text", ""))
        blocks = payload.get("blocks")

    if channel is None:
        channel = get_slack_channel()
    token = os.environ.get("SLACK_BOT_TOKEN")

    if not channel:
        logger.warning("Slack 채널이 설정되어 있지 않아 전송을 건너뜁니다.")
        return None

    if not token:
        logger.warning("SLACK_BOT_TOKEN 이 설정되지 않아 전송을 건너뜁니다.")
        return None

    client = get_slack_client(token)

    try:
        response = client.chat_postMessage(
            channel=channel,
            text=text or "Slack notification",
            blocks=blocks,
            thread_ts=thread_ts,
        )
        ts = response.get("ts")
        logger.info("Slack message sent via bot token (channel=%s, ts=%s)", channel, ts)
        return ts
    except SlackApiError as exc:  # pragma: no cover - 외부 API 호출 오류
        logger.error(
            "Slack API 호출 중 오류가 발생했습니다: %s",
            getattr(exc, "response", {}).get("error") or str(exc),
            exc_info=True,
        )
        return None
    except Exception:  # pragma: no cover - 방어적 처리
        logger.error("Slack 메시지 전송 중 알 수 없는 오류가 발생했습니다", exc_info=True)
        return None


def send_slack_message_v2(
    text: str,
    blocks: list[dict[str, Any]] | None = None,
    thread_ts: str | None = None,
    *,
    channel: str | None = None,
) -> str | None:
    """
    범용 슬랙 메시지 전송 함수 (WebClient 기반).
    메시지의 ts(timestamp)를 반환합니다.
    """
    return _send_slack_message_via_bot({"text": text, "blocks": blocks}, thread_ts=thread_ts, channel=channel)


def update_slack_message(
    text: str, *, channel: str, message_ts: str, blocks: list[dict[str, Any]] | None = None
) -> None:
    """봇이 보낸 메시지를 수정한다. 호출자가 슬랙 오류에 따라 처리할 수 있게 예외를 전달한다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("슬랙 메시지를 수정할 봇 토큰이 없습니다.")
    get_slack_client(token).chat_update(channel=channel, ts=message_ts, text=text, blocks=blocks)


def upload_slack_file(
    content: str,
    *,
    filename: str,
    title: str,
    initial_comment: str | None = None,
    channel: str | None = None,
) -> str | None:
    """텍스트 로그를 슬랙 채널에 파일로 업로드하고 permalink URL 을 반환한다 (`files:write` 권한 필요)."""
    target_channel = channel or get_slack_channel()
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not target_channel or not token:
        logger.warning("슬랙 파일 업로드를 건너뜁니다 (채널 또는 봇 토큰 없음).")
        return None

    raw = (content if content.strip() else "(내용 없음)\n").encode("utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        # files.getUploadURLExternal 은 form-encoded 파라미터를 사용한다.
        url_resp = _SLACK_HTTP_SESSION.post(
            "https://slack.com/api/files.getUploadURLExternal",
            headers=headers,
            data={"filename": filename, "length": len(raw)},
            timeout=30.0,
        )
        url_resp.raise_for_status()
        url_data = url_resp.json()
        if not url_data.get("ok"):
            logger.error("슬랙 업로드 URL 발급 실패: %s", url_data.get("error"))
            return None

        upload_resp = _SLACK_HTTP_SESSION.post(
            url_data["upload_url"],
            data=raw,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=30.0,
        )
        upload_resp.raise_for_status()

        complete_payload: dict[str, Any] = {
            "files": [{"id": url_data["file_id"], "title": title}],
            "channel_id": target_channel,
        }
        if initial_comment:
            complete_payload["initial_comment"] = initial_comment
        complete_resp = _SLACK_HTTP_SESSION.post(
            "https://slack.com/api/files.completeUploadExternal",
            headers=headers,
            json=complete_payload,
            timeout=30.0,
        )
        complete_resp.raise_for_status()
        complete_data = complete_resp.json()
        if not complete_data.get("ok"):
            logger.error("슬랙 파일 업로드 완료 실패: %s", complete_data.get("error"))
            return None
        files = complete_data.get("files") or []
        permalink = str(files[0].get("permalink") or "").strip() if files and isinstance(files[0], dict) else ""
        logger.info(
            "Slack 파일 업로드 완료 (channel=%s, filename=%s, permalink=%s)", target_channel, filename, permalink
        )
        return permalink or None
    except Exception:
        logger.error("Slack 파일 업로드 중 오류 발생", exc_info=True)
        return None


def _format_shares_for_country(quantity: Any) -> str:
    if not isinstance(quantity, (int, float)):
        return str(quantity)
    return f"{quantity:,.0f}"


def build_summary_line_from_summary_data(
    summary_data: dict[str, Any],
    money_formatter: callable = format_kr_money,
    *,
    use_html: bool,
    prefix: str | None = None,
    include_hold: bool = True,
) -> str:
    parts: list[str] = []

    if include_hold:
        held_count = summary_data.get("held_count")
        limit = summary_data.get("universe_count")
        if held_count is not None and limit is not None:
            try:
                parts.append(f"보유종목: {int(held_count)}/{int(limit)}")
            except (TypeError, ValueError):
                pass

    principal = money_formatter(float(summary_data.get("principal", 0.0) or 0.0))
    parts.append(f"원금: {principal}")

    parts.append(
        _format_return_with_amount(
            "일간",
            float(summary_data.get("daily_return_pct", 0.0) or 0.0),
            float(summary_data.get("daily_profit_loss", 0.0) or 0.0),
            money_formatter,
            use_html=use_html,
        )
    )
    parts.append(
        _format_return_with_amount(
            "평가",
            float(summary_data.get("eval_return_pct", 0.0) or 0.0),
            float(summary_data.get("eval_profit_loss", 0.0) or 0.0),
            money_formatter,
            use_html=use_html,
        )
    )
    parts.append(
        _format_return_with_amount(
            "누적",
            float(summary_data.get("cum_return_pct", 0.0) or 0.0),
            float(summary_data.get("cum_profit_loss", 0.0) or 0.0),
            money_formatter,
            use_html=use_html,
        )
    )

    cash = money_formatter(float(summary_data.get("total_cash", 0.0) or 0.0))
    equity = money_formatter(float(summary_data.get("total_equity", 0.0) or 0.0))
    parts.append(f"현금: {cash}")
    parts.append(f"평가금액: {equity}")

    body = " | ".join(parts)
    if prefix:
        return f"{prefix} {body}"
    return body


def build_summary_line_from_header(header_line: str, prefix: str | None = None) -> str:
    header_line_clean = header_line.split("<br>")[0]
    segments = [seg.strip() for seg in header_line_clean.split("|")]

    def _pick(label: str, default: str) -> str:
        for seg in segments:
            if label in seg:
                value = seg.split(":", 1)[1].strip()
                return f"{label}: {strip_html_tags(value)}"
        return default

    parts = [
        _pick("보유종목", "보유종목: -"),
        _pick("원금", "원금: -"),
        _pick("일간", "일간: +0.00%(0원)"),
        _pick("평가", "평가: +0.00%(0원)"),
        _pick("누적", "누적: +0.00%(0원)"),
        _pick("현금", "현금: 0원"),
        _pick("평가금액", "평가금액: 0원"),
    ]

    body = " | ".join(parts)
    if prefix:
        return f"{prefix} {body}"
    return body


def _format_return_with_amount(
    label: str,
    pct: float,
    amount: float,
    formatter: callable,
    *,
    use_html: bool,
) -> str:
    pct = float(pct or 0.0)
    amount = float(amount or 0.0)
    formatted_amount = formatter(amount)

    if use_html:
        color = "red" if pct > 0 else "blue" if pct < 0 else "black"
        return f'{label}: <span style="color:{color}">{pct:+.2f}%({formatted_amount})</span>'

    return f"{label}: {pct:+.2f}%({formatted_amount})"


__all__ = [
    "app_link",
    "build_summary_line_from_summary_data",
    "build_summary_line_from_header",
    "get_last_error",
    "send_slack_message",
    "send_slack_message_v2",
    "strip_html_tags",
]
