"""Slack notification helpers consolidated in one module."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import requests
import settings as global_settings

try:
    from croniter import croniter
    import pytz
except ImportError:  # pragma: no cover - optional dependency
    croniter = None
    pytz = None

from utils.account_registry import (
    get_account_file_settings,
    get_account_info,
)
from utils.schedule_config import get_country_schedule
from utils.data_loader import get_aud_to_krw_rate
from utils.db_manager import get_portfolio_snapshot
from utils.report import format_kr_money, render_table_eaw
from utils.stock_list_io import get_etfs

_LAST_ERROR: Optional[str] = None


# ---------------------------------------------------------------------------
# Slack webhook helpers
# ---------------------------------------------------------------------------


def get_slack_webhook_url(country: str, account: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Return (webhook_url, source_name) for the given account/country."""

    # 1. 계좌 파일에서 직접 지정한 웹훅이 있는지 확인합니다.
    if account:
        try:
            settings = get_account_file_settings(account)
            url = settings.get("slack_webhook_url")
            if url:
                return url, f"account:{account}"
        except SystemExit:
            pass

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
            "coin": "Asia/Seoul",
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
    except requests.exceptions.RequestException as exc:  # pragma: no cover - simple pass through
        _LAST_ERROR = str(exc)
        return False
    except Exception as exc:  # pragma: no cover - defensive
        _LAST_ERROR = str(exc)
        return False


def get_last_error() -> Optional[str]:
    return _LAST_ERROR


def send_slack_message_to_logs(message: str):
    webhook_url = os.environ.get("LOGS_SLACK_WEBHOOK")
    if webhook_url:
        log_message = f"📜 *[{global_settings.APP_TYPE}]*{message}"
        send_slack_message(log_message, webhook_url=webhook_url, webhook_name="LOGS_SLACK_WEBHOOK")


# def send_verbose_log_to_slack(message: str):
#     webhook_url = os.environ.get("VERBOSE_LOGS_SLACK_WEBHOOK")
#     if webhook_url:
#         log_message = f"📜 *[{global_settings.APP_TYPE}]*{message}"
#         send_slack_message(
#             log_message, webhook_url=webhook_url, webhook_name="VERBOSE_LOGS_SLACK_WEBHOOK"
#         )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def strip_html_tags(value: str) -> str:
    try:
        return re.sub(r"<[^>]+>", "", value)
    except Exception:  # pragma: no cover - defensive
        return value


def _format_shares_for_country(quantity: Any, country: str) -> str:
    if not isinstance(quantity, (int, float)):
        return str(quantity)
    if country == "coin":
        return f"{quantity:,.8f}".rstrip("0").rstrip(".")
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


# ---------------------------------------------------------------------------
# High level notifications
# ---------------------------------------------------------------------------


def send_summary_notification(
    country: str,
    account: str,
    report_date: datetime,
    duration: float,
    old_equity: float,
    summary_data: Optional[Dict[str, Any]] = None,
    header_line: Optional[str] = None,
    force_send: bool = False,
) -> None:
    """Send concise account summary to Slack."""

    from utils.transaction_manager import get_transactions_up_to_date

    if not force_send and not should_notify_on_schedule(country):
        return

    webhook_info = get_slack_webhook_url(country, account=account)
    if not webhook_info:
        return
    webhook_url, webhook_name = webhook_info

    # 항상 최신 평가금액을 확인합니다.
    new_snapshot = get_portfolio_snapshot(country, account=account)
    new_equity = float(new_snapshot.get("total_equity", 0.0)) if new_snapshot else 0.0

    # 필요 시 AUD -> KRW 환산
    account_currency = None
    try:
        account_info = get_account_info(account)
        account_currency = (account_info or {}).get("currency")
    except Exception:
        account_info = None

    aud_krw_rate = None
    if account_currency == "AUD":
        aud_krw_rate = get_aud_to_krw_rate()
        if aud_krw_rate:
            new_equity *= aud_krw_rate
            old_equity *= aud_krw_rate

    try:
        file_settings = get_account_file_settings(account)
        initial_capital_krw = float(file_settings.get("initial_capital_krw", 0))
    except SystemExit:
        initial_capital_krw = 0.0

    money_formatter = format_kr_money
    app_tag = getattr(global_settings, "APP_TYPE", "APP")
    app_prefix = app_tag if app_tag.startswith("[") and app_tag.endswith("]") else f"[{app_tag}]"
    summary_prefix = f"{app_prefix}[{country}/{account}]"

    if summary_data:
        new_equity = float(summary_data.get("total_equity", new_equity) or 0.0)
        summary_line = build_summary_line_from_summary_data(
            summary_data, money_formatter, use_html=False, prefix=summary_prefix
        )
    elif header_line:
        summary_line = build_summary_line_from_header(header_line, prefix=summary_prefix)
    else:
        parts = [f"금액: {money_formatter(new_equity)}"]

        if old_equity > 0 and new_equity > 0:
            day_ret_pct = ((new_equity / old_equity) - 1.0) * 100.0
            day_profit_loss = new_equity - old_equity
            parts.append(f"일간: {day_ret_pct:+.2f}%({money_formatter(day_profit_loss)})")

        if initial_capital_krw > 0:
            injections = get_transactions_up_to_date(
                country, account, report_date, "capital_injection"
            )
            withdrawals = get_transactions_up_to_date(
                country, account, report_date, "cash_withdrawal"
            )

            total_injections = sum(inj.get("amount", 0.0) for inj in injections)
            total_withdrawals = sum(wd.get("amount", 0.0) for wd in withdrawals)

            adjusted_capital_base = initial_capital_krw + total_injections
            adjusted_equity = new_equity + total_withdrawals

            if adjusted_capital_base > 0:
                eval_profit_loss = new_equity - adjusted_capital_base
                eval_ret_pct = (eval_profit_loss / adjusted_capital_base) * 100.0
                parts.append(f"평가: {eval_ret_pct:+.2f}%({money_formatter(eval_profit_loss)})")

                cum_ret_pct = ((adjusted_equity / adjusted_capital_base) - 1.0) * 100.0
                cum_profit_loss = adjusted_equity - adjusted_capital_base
                parts.append(f"누적: {cum_ret_pct:+.2f}%({money_formatter(cum_profit_loss)})")

        summary_line = f"{summary_prefix} " + " | ".join(parts)

    min_change_threshold = 0.5 if country != "aus" else 0.005
    change_segment = None
    if abs(new_equity - old_equity) >= min_change_threshold:
        diff = new_equity - old_equity
        change_label = "📈평가금액 증가" if diff > 0 else "📉평가금액 감소"

        if country == "aus" or abs(diff) >= 10_000:
            old_equity_str = money_formatter(old_equity)
            new_equity_str = money_formatter(new_equity)
            diff_str = f"{'+' if diff > 0 else ''}{money_formatter(diff)}"
        else:
            old_equity_str = f"{int(round(old_equity)):,}원"
            new_equity_str = f"{int(round(new_equity)):,}원"
            diff_int = int(round(diff))
            diff_str = (
                f"{'+' if diff_int > 0 else ''}{diff_int:,}원" if diff_int else f"{diff:+.2f}원"
            )

        change_segment = f"{change_label}: {old_equity_str} => {new_equity_str} ({diff_str})"

    final_message = summary_line
    if change_segment:
        final_message = f"{final_message} | {change_segment}"

    slug = f"summary:{country}/{account}"
    send_slack_message(final_message, webhook_url=webhook_url, webhook_name=slug)
    send_slack_message_to_logs(final_message)


def send_detailed_signal_notification(
    country: str,
    account: str,
    header_line: str,
    headers: List[str],
    rows_sorted: List[List[Any]],
    *,
    decision_config: Dict[str, Any],
    extra_lines: Optional[List[str]] = None,
    force_send: bool = False,
    save_to_path: Optional[str] = None,
) -> bool:
    """Render and send detailed signal table to Slack."""

    webhook_info = get_slack_webhook_url(country, account=account)
    if not webhook_info:
        return False

    if not force_send and not should_notify_on_schedule(country):
        return False

    webhook_url, webhook_name = webhook_info

    idx_ticker = headers.index("티커")
    idx_state = headers.index("상태") if "상태" in headers else None
    idx_ret = (
        headers.index("누적수익률")
        if "누적수익률" in headers
        else (headers.index("일간수익률") if "일간수익률" in headers else None)
    )
    idx_score = headers.index("점수") if "점수" in headers else None

    try:
        etfs = get_etfs(country) or []
        name_map = {str(s.get("ticker") or "").upper(): str(s.get("name") or "") for s in etfs}
    except Exception:
        name_map = {}

    if country == "aus":
        name_map["IS"] = "International Shares"

    display_rows: List[Dict[str, str]] = []
    width_tracker = {
        "name": 0,
        "return": 0,
        "score": 0,
    }

    for row in rows_sorted:
        try:
            num_part = f"[{row[0]}]"
            tkr = str(row[idx_ticker])
            name = name_map.get(tkr.upper(), "")
            if country == "aus" and tkr.upper() == "IS":
                name_part = name
            else:
                name_part = f"{name}({tkr})" if name else tkr
            full_name_part = f"{num_part} {name_part}"

            state = str(row[idx_state]) if idx_state is not None and idx_state < len(row) else ""
            return_col = ""
            if idx_ret is not None:
                ret_value = row[idx_ret]
                if isinstance(ret_value, (int, float)) and abs(ret_value) > 0.001:
                    return_col = f"수익 {ret_value:+.2f}%,"

            score_col = ""
            if idx_score is not None:
                score_value = row[idx_score]
                if isinstance(score_value, (int, float)):
                    score_col = f"점수 {float(score_value):.1f}"

            try:
                score_value_float = float(score_value)
            except (TypeError, ValueError):
                score_value_float = float("nan")

            note_text = ""
            if isinstance(row, (list, tuple)) and row:
                try:
                    note_text = str(row[-1]).strip()
                except Exception:
                    note_text = ""

            display_rows.append(
                {
                    "name": full_name_part,
                    "state": state,
                    "return": return_col,
                    "score": score_col,
                    "score_value": score_value_float,
                    "note": note_text,
                }
            )

            width_tracker["name"] = max(width_tracker["name"], len(full_name_part))
            width_tracker["return"] = max(width_tracker["return"], len(return_col))
            width_tracker["score"] = max(width_tracker["score"], len(score_col))
        except Exception:
            continue

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for parts in display_rows:
        grouped.setdefault(parts["state"], []).append(parts)

    lines: List[str] = []
    for state, items in sorted(
        grouped.items(),
        key=lambda item: decision_config.get(item[0], {"order": 99}).get("order", 99),
    ):
        trimmed_suffix = ""
        if state == "WAIT" and items:

            def _score_value(part):
                val = part.get("score_value")
                if isinstance(val, float) and math.isnan(val):
                    return float("-inf")
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return float("-inf")

            items = sorted(items, key=_score_value, reverse=True)
            if len(items) > 10:
                items = items[:10]
                trimmed_suffix = " (TOP 10)"

        config = decision_config.get(state)
        if not config:
            display_name = f"<{state}>({state})"
            show_slack = True
        else:
            display_name = f"{config['display_name']}({state})"
            show_slack = config.get("show_slack", True)

        if state == "WAIT":
            show_slack = True

        if not show_slack:
            continue

        if not items:
            continue

        lines.append(display_name + trimmed_suffix)
        show_return_col = state in {"HOLD", "BUY", "BUY_REPLACE"}
        for parts in items:
            name_part = parts["name"].ljust(width_tracker["name"])
            score_part = parts["score"].ljust(width_tracker["score"])
            return_text = parts["return"].ljust(width_tracker["return"]) if show_return_col else ""
            note_text = parts.get("note", "")
            line = f"{name_part}  {return_text} {score_part}".rstrip()
            if note_text:
                line = f"{line}  {note_text}"
            lines.append(line)
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    app_tag = getattr(global_settings, "APP_TYPE", "APP")
    title = f"[{app_tag}][{country}/{account}] 종목상세"
    message_header_parts = [title]
    if extra_lines:
        message_header_parts.extend(extra_lines)
    message_header = "\n".join(message_header_parts)

    has_recommendation = any(
        decision_config.get(state, {}).get("is_recommendation", False) for state in grouped.keys()
    )
    # If the account is configured not to need signals, avoid channel-wide mention
    try:
        need_signal = bool((get_account_info(account) or {}).get("need_signal", True))
    except Exception:
        need_signal = True
    slack_prefix = "<!channel>\n" if has_recommendation and need_signal else ""

    # 파일 저장용 렌더링 텍스트 (백테스트 스타일)과 Slack 메시지 본문(코드블럭) 분리 구성
    if not lines:
        rendered_text = message_header
    else:
        rendered_text = message_header + "\n\n" + "```\n" + "\n".join(lines) + "\n```"

    # 요청 시 파일로 저장: 백테스트 스타일 테이블 (render_table_eaw)
    if save_to_path:
        try:
            # 1) 헤더 요약 (웹/슬랙 헤더에서 텍스트 추출)
            summary_line = build_summary_line_from_header(header_line)

            # 2) 정렬(aligns) 정의: 숫자/퍼센트/금액은 right, 그 외 left
            numeric_like = {
                "#",
                "현재가",
                "일간수익률",
                "보유수량",
                "금액",
                "누적수익률",
                "비중",
                "고점대비",
                "점수",
                "지속",
            }
            aligns = ["right" if (str(h) in numeric_like) else "left" for h in headers]

            # 3) 퍼센트 표시 자릿수 설정 로드 및 적용
            def _load_display_precision() -> Dict[str, int]:
                try:
                    root = Path(__file__).resolve().parent.parent  # project root
                    cfg_path = root / "data" / "settings" / "precision.json"
                    if not cfg_path.exists():
                        return {
                            "daily_return_pct": 2,
                            "cum_return_pct": 2,
                            "weight_pct": 2,
                        }
                    import json

                    with open(cfg_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp) or {}
                    prec = data.get("common") or {}
                    return {
                        "daily_return_pct": int(prec.get("daily_return_pct", 2)),
                        "cum_return_pct": int(prec.get("cum_return_pct", 2)),
                        "weight_pct": int(prec.get("weight_pct", 2)),
                    }
                except Exception:
                    return {
                        "daily_return_pct": 2,
                        "cum_return_pct": 2,
                        "weight_pct": 2,
                    }

            prec = _load_display_precision()
            p_daily = max(0, int(prec.get("daily_return_pct", 2)))
            p_cum = max(0, int(prec.get("cum_return_pct", 2)))
            p_w = max(0, int(prec.get("weight_pct", 2)))

            # 헤더 인덱스 찾기
            idx_day = headers.index("일간수익률") if "일간수익률" in headers else None
            idx_cum = headers.index("누적수익률") if "누적수익률" in headers else None
            idx_w = headers.index("비중") if "비중" in headers else None
            idx_sh = headers.index("보유수량") if "보유수량" in headers else None
            idx_px = headers.index("현재가") if "현재가" in headers else None
            idx_amt = headers.index("금액") if "금액" in headers else None

            # 국가별 수량 정밀도 로더
            def _load_country_precision(country_key: str) -> Dict[str, Any]:
                try:
                    root = Path(__file__).resolve().parent.parent
                    cfg_path = root / "data" / "settings" / "precision.json"
                    import json

                    with open(cfg_path, "r", encoding="utf-8") as fp:
                        data = json.load(fp) or {}
                    c = (data.get("country") or {}).get(country_key)
                    return c if isinstance(c, dict) else {}
                except Exception:
                    return {}

            cprec = _load_country_precision(country)
            qty_p = int(cprec.get("qty_precision", 0)) if isinstance(cprec, dict) else 0
            amt_p = int(cprec.get("amt_precision", 0)) if isinstance(cprec, dict) else 0

            # 4) 문자열 변환 + 정밀도 적용된 표 데이터 생성
            formatted_rows: List[List[str]] = []
            for row in rows_sorted:
                fr = []
                for j, c in enumerate(row):
                    val = c
                    if (idx_day is not None) and (j == idx_day) and isinstance(val, (int, float)):
                        fr.append(("{:+." + str(p_daily) + "f}%").format(float(val)))
                    elif (idx_cum is not None) and (j == idx_cum) and isinstance(val, (int, float)):
                        fr.append(("{:+." + str(p_cum) + "f}%").format(float(val)))
                    elif (idx_w is not None) and (j == idx_w) and isinstance(val, (int, float)):
                        fr.append(("{:." + str(p_w) + "f}%").format(float(val)))
                    elif (idx_sh is not None) and (j == idx_sh) and isinstance(val, (int, float)):
                        if qty_p > 0:
                            s = (
                                ("{:." + str(qty_p) + "f}")
                                .format(float(val))
                                .rstrip("0")
                                .rstrip(".")
                            )
                            fr.append(s if s != "" else "0")
                        else:
                            fr.append(f"{int(round(float(val))):,d}")
                    elif (idx_px is not None) and (j == idx_px) and isinstance(val, (int, float)):
                        fmt = ("{:, ." + str(amt_p) + "f}") if amt_p > 0 else "{:, .0f}"
                        fmt = fmt.replace(" ", "")
                        fr.append(fmt.format(float(val)))
                    elif (idx_amt is not None) and (j == idx_amt) and isinstance(val, (int, float)):
                        fmt = ("{:, ." + str(amt_p) + "f}") if amt_p > 0 else "{:, .0f}"
                        fmt = fmt.replace(" ", "")
                        fr.append(fmt.format(float(val)))
                    else:
                        fr.append("-" if (val is None) else str(val))
                formatted_rows.append(fr)

            table_lines = render_table_eaw(headers, formatted_rows, aligns)

            # 4) 추가 라인(경고/노트 등) 포함
            extras = "\n".join(extra_lines or [])
            backtest_text = summary_line + "\n\n" + "\n".join(table_lines)
            if extras:
                backtest_text += "\n\n" + extras

            import os

            os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
            with open(save_to_path, "w", encoding="utf-8") as fp:
                fp.write(backtest_text)
        except Exception:
            # 파일 저장 실패는 슬랙 전송에 영향 주지 않음
            pass

    # Slack 전송
    if not lines:
        return send_slack_message(
            slack_prefix + message_header, webhook_url=webhook_url, webhook_name=webhook_name
        )

    message = rendered_text
    return send_slack_message(
        slack_prefix + message, webhook_url=webhook_url, webhook_name=webhook_name
    )


__all__ = [
    "build_summary_line_from_summary_data",
    "build_summary_line_from_header",
    "get_last_error",
    "get_slack_webhook_url",
    "send_detailed_signal_notification",
    "send_slack_message_to_logs",
    "send_slack_message",
    "send_summary_notification",
    # "send_verbose_log_to_slack",
    "should_notify_on_schedule",
    "strip_html_tags",
]
