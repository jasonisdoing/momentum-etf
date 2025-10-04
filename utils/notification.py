"""Slack notification helpers consolidated in one module."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import requests
import settings as global_settings

try:
    from slack_sdk import WebClient  # type: ignore
    from slack_sdk.errors import SlackApiError  # type: ignore
except Exception:  # pragma: no cover - 선택적 의존성 처리
    WebClient = None  # type: ignore
    SlackApiError = Exception  # type: ignore

try:
    from croniter import croniter
    import pytz
except ImportError:  # pragma: no cover - 선택적 의존성 처리
    croniter = None
    pytz = None

from utils.account_registry import get_account_settings
from utils.schedule_config import get_country_schedule
from utils.report import format_kr_money

_LAST_ERROR: Optional[str] = None
APP_LABEL = getattr(global_settings, "APP_TYPE", "APP")


# ---------------------------------------------------------------------------
# 슬랙 웹훅 관련 헬퍼
# ---------------------------------------------------------------------------


def get_slack_webhook_url(country: str) -> Optional[Tuple[str, str]]:
    """Return (webhook_url, source_name) for the given country.

    Args:
        country: 국가 코드 (예: 'kor', 'aus')

    Returns:
        (웹훅 URL, 소스 이름) 튜플 또는 사용 가능한 웹훅이 없으면 None
    """
    # 1. 국가 설정에서 웹훅 URL 가져오기
    country_settings = get_account_settings(country)
    if country_settings:
        url = country_settings.get("slack_webhook_url")
        if url:
            return url, f"country:{country}"

    # 2. 환경 변수 (국가 단위) 확인
    env_var_name = f"{country.upper()}_SLACK_WEBHOOK"
    url = os.environ.get(env_var_name)
    if url:
        return url, env_var_name

    return None


def should_notify_on_schedule(country: str) -> bool:
    """Check notification schedule (CRON)."""

    if not croniter or not pytz:
        return True

    notify_cron_env = f"NOTIFY_{country.upper()}_CRON"
    cron_schedule = os.environ.get(notify_cron_env)
    config = get_country_schedule(country)
    if cron_schedule is None:
        cron_schedule = config.get("notify_cron")

    if not cron_schedule:
        return True

    tz_env = f"SCHEDULE_{country.upper()}_TZ"
    default_tz = (
        config.get("notify_timezone")
        or config.get("timezone")
        or {
            "kor": "Asia/Seoul",
            "aus": "Australia/Sydney",
        }.get(country, "Asia/Seoul")
    )
    tz_str = os.environ.get(tz_env, default_tz)

    try:
        local_tz = pytz.timezone(tz_str)
        now = datetime.now(local_tz)
        return croniter.match(cron_schedule, now)
    except Exception:
        return True


def send_slack_message(
    text: str,
    blocks: Optional[List[dict]] = None,
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
        payload = {"text": text.replace("<pre>", "```").replace("</pre>", "```")}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        if response.text == "ok":
            print(f"[SLACK] 메시지 전송 - {webhook_name or 'Unknown Webhook'}")
            return True
        _LAST_ERROR = response.text or "unknown"
        return False
    except requests.exceptions.RequestException as exc:  # pragma: no cover - 단순 위임 예외 처리
        _LAST_ERROR = str(exc)
        return False
    except Exception as exc:  # pragma: no cover - 방어적 처리
        _LAST_ERROR = str(exc)
        return False


def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def send_slack_message_to_logs(message: str):
    webhook_url = os.environ.get("LOGS_SLACK_WEBHOOK")
    if webhook_url:
        log_message = f"📜 *[{global_settings.APP_TYPE}]*{message}"
        send_slack_message(log_message, webhook_url=webhook_url, webhook_name="LOGS_SLACK_WEBHOOK")


def _upload_file_to_slack(
    *, channel: Optional[str], file_path: Path, title: str, initial_comment: Optional[str] = None
) -> bool:
    """Upload a file to Slack using the Web API if a bot token is available."""

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        print("[SLACK] 파일 업로드 생략 - SLACK_BOT_TOKEN 미설정")
        return False
    if WebClient is None:
        print("[SLACK] 파일 업로드 생략 - slack_sdk 미설치")
        return False

    file_exists = file_path.exists() and file_path.is_file()
    if not file_exists:
        print(f"[SLACK] 파일 업로드 생략 - 파일 없음: {file_path}")
        return False

    error_message: Optional[str] = None

    try:
        client = WebClient(token=token)
        client.files_upload_v2(
            channel=channel,
            file=str(file_path),
            title=title,
            initial_comment=initial_comment,
        )
        print(f"[SLACK] 파일 업로드 성공 - channel={channel} file={file_path.name}")
        return True
    except SlackApiError as exc:  # pragma: no cover - 외부 API 의존 처리
        error_message = getattr(exc, "response", {}).get("error") or str(exc)
    except Exception as exc:  # pragma: no cover - 방어적 처리
        error_message = str(exc)

    if error_message:
        print(f"[SLACK] 파일 업로드 실패 - channel={channel} file={file_path.name} reason={error_message}")

    return False


# def send_verbose_log_to_slack(message: str):
#     webhook_url = os.environ.get("VERBOSE_LOGS_SLACK_WEBHOOK")
#     if webhook_url:
#         log_message = f"📜 *[{global_settings.APP_TYPE}]*{message}"
#         send_slack_message(
#             log_message, webhook_url=webhook_url, webhook_name="VERBOSE_LOGS_SLACK_WEBHOOK"
#         )


# ---------------------------------------------------------------------------
# 포맷팅 보조 함수
# ---------------------------------------------------------------------------


def strip_html_tags(value: str) -> str:
    try:
        return re.sub(r"<[^>]+>", "", value)
    except Exception:  # pragma: no cover - 방어적 처리
        return value


def compose_recommendation_slack_message(
    account_id: str,
    report: Any,
    *,
    duration: float,
    force_notify: bool = False,
) -> str:
    """Compose a minimal Slack message with dashboard link for recommendations."""

    base_date = getattr(report, "base_date", None)
    if hasattr(base_date, "strftime"):
        try:
            base_date_str = base_date.strftime("%Y-%m-%d")
        except Exception:  # pragma: no cover - 방어적 처리
            base_date_str = str(base_date)
    elif base_date is not None:
        base_date_str = str(base_date)
    else:
        base_date_str = "N/A"

    headline = f"[{APP_LABEL}] 종목 추천 정보가 갱신되었습니다. ({base_date_str})"
    account_norm = (account_id or "").strip().lower()
    dashboard_url = (
        f"http://localhost:8501/{account_norm}" if account_norm else "http://localhost:8501/"
    )

    recommendations = list(getattr(report, "recommendations", []) or [])
    decision_config = getattr(report, "decision_config", {}) or {}

    lines = [headline, f"생성시간: {duration:.1f}초"]

    state_counter: Counter[str] = Counter()
    if isinstance(decision_config, dict):
        for item in recommendations:
            state = str(item.get("state") or "").upper()
            cfg = decision_config.get(state)
            if cfg and cfg.get("is_recommendation"):
                state_counter[state] += 1

    def _state_order(state: str) -> int:
        cfg = decision_config.get(state)
        if isinstance(cfg, dict):
            try:
                return int(cfg.get("order", 99))
            except (TypeError, ValueError):
                return 99
        return 99

    state_lines: List[str] = []
    for state, count in sorted(
        state_counter.items(), key=lambda pair: (_state_order(pair[0]), pair[0])
    ):
        state_lines.append(f"{state}: {count}개")

    if state_lines:
        lines.extend(state_lines)

    lines.append(dashboard_url)

    body = "\n".join(lines)
    if state_lines:
        return "<!channel>\n" + body
    return body


def _format_shares_for_country(quantity: Any, country: str) -> str:
    if not isinstance(quantity, (int, float)):
        return str(quantity)
    return f"{quantity:,.0f}"


def build_summary_line_from_summary_data(
    summary_data: Dict[str, Any],
    money_formatter: callable = format_kr_money,
    *,
    use_html: bool,
    prefix: Optional[str] = None,
    include_hold: bool = True,
) -> str:
    parts: List[str] = []

    if include_hold:
        held_count = summary_data.get("held_count")
        portfolio_topn = summary_data.get("portfolio_topn")
        if held_count is not None and portfolio_topn is not None:
            try:
                parts.append(f"보유종목: {int(held_count)}/{int(portfolio_topn)}")
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


def build_summary_line_from_header(header_line: str, prefix: Optional[str] = None) -> str:
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
    "compose_recommendation_slack_message",
    "build_summary_line_from_summary_data",
    "build_summary_line_from_header",
    "get_last_error",
    "get_slack_webhook_url",
    "send_slack_message_to_logs",
    "send_slack_message",
    # "send_verbose_log_to_slack",
    "should_notify_on_schedule",
    "strip_html_tags",
]
