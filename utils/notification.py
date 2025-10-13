"""Slack notification helpers consolidated in one module."""

from __future__ import annotations

import os
import re
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter
from numbers import Number

import textwrap

import requests

try:
    import matplotlib

    matplotlib.use("Agg")  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - 선택적 의존성 처리
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

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
from utils.logger import get_app_logger
from utils.settings_loader import get_account_slack_channel
from dotenv import load_dotenv
from utils.cron_utils import normalize_cron_weekdays

load_dotenv()
APP_VERSION = "2025-10-13-11"
APP_LABEL = os.environ.get("APP_TYPE", f"APP-{APP_VERSION}")

_LAST_ERROR: Optional[str] = None
logger = get_app_logger()


_DEFAULT_REPORT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "cache" / "slack_reports"


# ---------------------------------------------------------------------------
# 슬랙 웹훅 관련 헬퍼
# ---------------------------------------------------------------------------


def should_notify_on_schedule(country: str) -> bool:
    """Check notification schedule (CRON)."""

    if not croniter or not pytz:
        return True

    config = get_country_schedule(country)
    cron_schedule = config.get("notify_cron")

    if not cron_schedule:
        return True

    cron_schedule = normalize_cron_weekdays(cron_schedule, target="croniter")

    tz_str = "Asia/Seoul"

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


def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def _upload_file_to_slack(*, channel: Optional[str], file_path: Path, title: str, initial_comment: Optional[str] = None) -> bool:
    """Upload a file to Slack using the Web API if a bot token is available."""

    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        logger.info("[SLACK] 파일 업로드 생략 - SLACK_BOT_TOKEN 미설정")
        return False
    if WebClient is None:
        logger.info("[SLACK] 파일 업로드 생략 - slack_sdk 미설치")
        return False

    file_exists = file_path.exists() and file_path.is_file()
    if not file_exists:
        logger.info("[SLACK] 파일 업로드 생략 - 파일 없음: %s", file_path)
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
        logger.info("[SLACK] 파일 업로드 성공 - channel=%s file=%s", channel, file_path.name)
        return True
    except SlackApiError as exc:  # pragma: no cover - 외부 API 의존 처리
        error_message = getattr(exc, "response", {}).get("error") or str(exc)
    except Exception as exc:  # pragma: no cover - 방어적 처리
        error_message = str(exc)

    if error_message:
        logger.warning(
            "[SLACK] 파일 업로드 실패 - channel=%s file=%s reason=%s",
            channel,
            file_path.name,
            error_message,
        )

    return False


# def send_verbose_log_to_slack(message: str):
#     webhook_url = os.environ.get("VERBOSE_LOGS_SLACK_WEBHOOK")
#     if webhook_url:
#         log_message = f"📜 *[{APP_LABEL}]*{message}"
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
) -> dict[str, Any]:
    """Compose Slack text and Block Kit payload for recommendation updates."""

    account_norm = (account_id or "").strip().lower()
    account_settings: dict[str, Any] | None = None
    try:
        account_settings = get_account_settings(account_norm)
    except Exception:
        account_settings = None

    account_label = str((account_settings or {}).get("name")) if account_settings and (account_settings or {}).get("name") else account_norm.upper()

    base_date = getattr(report, "base_date", None)
    if hasattr(base_date, "strftime"):
        try:
            base_date_str = base_date.strftime("%Y-%m-%d")
        except Exception:
            base_date_str = str(base_date)
    elif base_date is not None:
        base_date_str = str(base_date)
    else:
        base_date_str = "N/A"

    recommendations = list(getattr(report, "recommendations", []) or [])
    decision_config = getattr(report, "decision_config", {}) or {}
    summary_data = getattr(report, "summary_data", None)

    held_count: int | None = None
    portfolio_topn: int | None = None
    if isinstance(summary_data, dict):
        held_raw = summary_data.get("held_count")
        topn_raw = summary_data.get("portfolio_topn")
        try:
            held_count = int(held_raw) if held_raw is not None else None
        except (TypeError, ValueError):
            held_count = None
        try:
            portfolio_topn = int(topn_raw) if topn_raw is not None else None
        except (TypeError, ValueError):
            portfolio_topn = None

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

    ordered_states = [(state, count) for state, count in sorted(state_counter.items(), key=lambda pair: (_state_order(pair[0]), pair[0]))]

    top_rows: list[str] = []
    for item in recommendations[:]:
        ticker = str(item.get("ticker") or "-")
        name = str(item.get("name") or ticker)
        state = str(item.get("state") or "-").upper()
        score = item.get("score")
        score_str = f"{score:.1f}" if isinstance(score, (int, float)) else "-"
        phrase = str(item.get("phrase") or "").strip()
        phrase_display = phrase if phrase else "-"
        top_rows.append(f"• {name}({ticker}) - {state} - 점수 {score_str} - {phrase_display}")

    headline = f"{account_label} 추천 정보가 갱신되었습니다. ({base_date_str})"
    app_prefix = f"[{APP_LABEL}] " if APP_LABEL else ""

    def _format_hold_ratio(held: int | None, topn: int | None) -> str:
        held_str = str(held) if held is not None else "?"
        topn_str = str(topn) if topn is not None else "?"
        return f"{held_str}/{topn_str}"

    if held_count is None:
        held_count = sum(1 for item in recommendations if str(item.get("state") or "").upper() == "HOLD")
    if portfolio_topn is None:
        topn_candidates = [
            getattr(report, "portfolio_topn", None),
            (account_settings or {}).get("portfolio_topn") if account_settings else None,
            (((account_settings or {}).get("strategy", {}) or {}).get("tuning", {}).get("PORTFOLIO_TOPN") if account_settings else None),
            (((account_settings or {}).get("strategy", {}) or {}).get("static", {}).get("PORTFOLIO_TOPN") if account_settings else None),
        ]
        for candidate in topn_candidates:
            try:
                portfolio_topn = int(candidate)
                break
            except (TypeError, ValueError, AttributeError):
                portfolio_topn = None

    lines = [
        app_prefix + headline,
        f"생성시간: {duration:.1f}초",
        f"보유종목: {_format_hold_ratio(held_count, portfolio_topn)}",
    ]
    if ordered_states:
        lines.extend([f"{state}: {count}개" for state, count in ordered_states])
    fallback_text = "\n".join(lines)

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"[{APP_LABEL}] {account_label}",
                "emoji": True,
            },
        }
    ]

    fields: list[dict[str, str]] = [
        {"type": "mrkdwn", "text": f"*기준일*: {base_date_str}"},
        {"type": "mrkdwn", "text": f"*소요시간*: {duration:.1f}초"},
        {"type": "mrkdwn", "text": f"*보유*: {_format_hold_ratio(held_count, portfolio_topn)}"},
    ]

    # 튜닝 파라미터 표시
    strategy_params = getattr(report, "strategy_params", {})
    if isinstance(strategy_params, dict):
        ma_period = strategy_params.get("MA_PERIOD")
        topn = strategy_params.get("PORTFOLIO_TOPN")
        replace_threshold = strategy_params.get("REPLACE_SCORE_THRESHOLD")

        params_str_parts = []
        if ma_period is not None:
            params_str_parts.append(f"MA: {ma_period}")
        if topn is not None:
            params_str_parts.append(f"TopN: {topn}")
        if replace_threshold is not None:
            params_str_parts.append(f"교체점수: {replace_threshold}")
        fields.append({"type": "mrkdwn", "text": f"*전략*: {', '.join(params_str_parts)}"})

    if ordered_states:
        state_lines = [f"{state}: {count}개" for state, count in ordered_states]
        fields.append({"type": "mrkdwn", "text": "*상태 요약*:\n" + "\n".join(state_lines)})

    blocks.append({"type": "section", "fields": fields})

    add_channel_mention = bool(ordered_states)

    if add_channel_mention:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "<!channel> 알림"}]})
        fallback_text = "<!channel>\n" + fallback_text

    payload: dict[str, Any] = {"text": fallback_text, "blocks": blocks}

    image_path = _generate_recommendation_report_image(
        report,
        title=f"{account_label} 추천 결과 ({base_date_str})",
    )
    if image_path:
        payload["report_image_path"] = str(image_path)
        # comment_lines = [
        #     textwrap.shorten(row, width=80, placeholder="…")
        #     for row in (top_rows[:3] if top_rows else [])
        # ]
        # comment_header = f"{account_label} 추천 결과 이미지 ({base_date_str})"
        # if comment_lines:
        #     payload["report_image_comment"] = comment_header + "\n" + "\n".join(comment_lines)
        # else:
        #     payload["report_image_comment"] = comment_header
        payload["report_image_title"] = f"{account_label} 추천 리포트"

    return payload


def _generate_recommendation_report_image(
    report: Any,
    *,
    title: str,
    max_rows: int = 20,
) -> Optional[Path]:
    if plt is None:
        logger.info("[SLACK] 추천 이미지 생략 - matplotlib 미설치")
        return None

    detail_headers = getattr(report, "detail_headers", None)
    detail_rows = getattr(report, "detail_rows", None)
    if not detail_headers or not detail_rows:
        logger.info("[SLACK] 추천 이미지 생략 - detail_rows 데이터 없음")
        return None

    display_headers = list(detail_headers)
    display_indexes = list(range(len(detail_headers)))

    if len(display_headers) < 3:
        logger.info("[SLACK] 추천 이미지 생략 - 표시할 컬럼 수 부족")
        return None

    cell_rows: list[list[str]] = []
    for row in detail_rows[:max_rows]:
        formatted_row = []
        for idx, label in zip(display_indexes, display_headers):
            value = row[idx] if idx < len(row) else None
            formatted_row.append(_format_image_cell_value(label, value))
        cell_rows.append(formatted_row)

    if not cell_rows:
        logger.info("[SLACK] 추천 이미지 생략 - 표시할 행 없음")
        return None

    fig_width = max(8.0, 1.2 * len(display_headers))
    fig_height = max(2.5, 0.45 * len(cell_rows) + 1.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    try:
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)

        table = ax.table(
            cellText=cell_rows,
            colLabels=display_headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)

        for col in range(len(display_headers)):
            try:
                table.auto_set_column_width(col)
            except Exception:
                pass

        for (row_idx, _col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d0d0")
            if row_idx == 0:
                cell.set_facecolor("#f2f2f2")
                cell.set_text_props(weight="bold")

        # Reduce surrounding whitespace while keeping the table readable.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.05)

        output_dir = Path(os.environ.get("SLACK_REPORT_IMAGE_DIR", _DEFAULT_REPORT_IMAGE_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)

        account_id = str(getattr(report, "account_id", "account") or "account")
        account_safe = re.sub(r"[^0-9A-Za-z_-]", "_", account_id)

        base_date = getattr(report, "base_date", None)
        if hasattr(base_date, "strftime"):
            base_date_str = base_date.strftime("%Y%m%d")
        else:
            base_date_str = str(base_date or "unknown")

        timestamp = datetime.now().strftime("%H%M%S")
        file_path = output_dir / f"recommend_{account_safe}_{base_date_str}_{timestamp}.png"

        fig.tight_layout(pad=0.6)
        fig.savefig(file_path, dpi=150)
        logger.info("[SLACK] 추천 이미지 생성 - %s", file_path)
        return file_path
    except Exception as exc:  # pragma: no cover - 시각화 오류 방어
        logger.warning("[SLACK] 추천 이미지 생성 실패: %s", exc)
        return None
    finally:
        plt.close(fig)


def _format_image_cell_value(column_label: str, value: Any) -> str:
    if value is None:
        return "-"

    if isinstance(value, str):
        clean = value.strip()
        if not clean:
            return "-"
        if column_label == "문구":
            return textwrap.shorten(clean, width=40, placeholder="…")
        return clean

    if isinstance(value, Number):
        if isinstance(value, float) and math.isnan(value):
            return "-"

        if column_label == "순위":
            return str(int(round(value)))
        if column_label == "점수":
            return f"{float(value):.2f}"
        if column_label == "현재가":
            return f"{float(value):,.2f}"
        if "%" in column_label:
            return f"{float(value):+.2f}%"
        return str(value)

    return str(value)


def send_recommendation_slack_notification(
    account_id: str,
    payload: dict[str, Any] | str,
) -> bool:
    """전달받은 페이로드를 슬랙으로 전송하고 필요 시 이미지 파일을 첨부합니다."""

    image_path_obj: Path | None = None
    image_comment: Optional[str] = None
    image_title: Optional[str] = None

    if isinstance(payload, str):
        text = payload
        blocks: list[dict[str, Any]] | None = None
    else:
        text = str(payload.get("text", ""))
        blocks = payload.get("blocks")
        image_raw = payload.get("report_image_path")
        if isinstance(image_raw, str) and image_raw:
            image_path_obj = Path(image_raw)
        image_comment_raw = payload.get("report_image_comment")
        if isinstance(image_comment_raw, str) and image_comment_raw.strip():
            image_comment = image_comment_raw.strip()
        image_title_raw = payload.get("report_image_title")
        if isinstance(image_title_raw, str) and image_title_raw.strip():
            image_title = image_title_raw.strip()

    channel = get_account_slack_channel(account_id)
    token = os.environ.get("SLACK_BOT_TOKEN")

    if not channel:
        logger.warning("Slack 채널이 설정되어 있지 않아 전송을 건너뜁니다 (account=%s)", account_id)
        return False

    if not token:
        logger.warning("SLACK_BOT_TOKEN 이 설정되지 않아 전송을 건너뜁니다 (account=%s)", account_id)
        return False

    if WebClient is None:
        logger.warning("slack_sdk 가 설치되어 있지 않아 슬랙 전송을 건너뜁니다 (account=%s)", account_id)
        return False

    client = WebClient(token=token)

    try:
        client.chat_postMessage(
            channel=channel,
            text=text or "Slack notification",
            blocks=blocks,
        )
        logger.info(
            "Slack message sent via bot token for account=%s (channel=%s)",
            account_id,
            channel,
        )
    except SlackApiError as exc:  # pragma: no cover - 외부 API 호출 오류
        logger.error(
            "Slack API 호출 중 오류가 발생했습니다 (account=%s): %s",
            account_id,
            getattr(exc, "response", {}).get("error") or str(exc),
            exc_info=True,
        )
        return False
    except Exception:  # pragma: no cover - 방어적 처리
        logger.error(
            "Slack 메시지 전송 중 알 수 없는 오류가 발생했습니다 (account=%s)",
            account_id,
            exc_info=True,
        )
        return False

    if image_path_obj:
        if image_path_obj.exists():
            uploaded = _upload_file_to_slack(
                channel=channel,
                file_path=image_path_obj,
                title=image_title or image_path_obj.name,
                initial_comment=image_comment,
            )
            if not uploaded:
                logger.warning(
                    "Slack 이미지 업로드 실패 (account=%s, file=%s)",
                    account_id,
                    image_path_obj.name,
                )
        else:
            logger.warning(
                "Slack 이미지 파일을 찾을 수 없습니다 (account=%s, path=%s)",
                account_id,
                image_path_obj,
            )

    return True


def _format_shares_for_country(quantity: Any) -> str:
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
    "send_slack_message",
    "send_recommendation_slack_notification",
    "should_notify_on_schedule",
    "strip_html_tags",
]
