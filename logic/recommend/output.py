"""Helpers for CLI recommendation output."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.notification import strip_html_tags
from utils.report import render_table_eaw
from utils.logger import get_app_logger

logger = get_app_logger()


def invoke_account_pipeline(account_id: str, *, date_str: str | None) -> List[Dict[str, Any]]:
    """Run the recommendation pipeline and return raw recommendation rows."""

    from logic.recommend import generate_recommendation_report

    try:
        result = generate_recommendation_report(account_id=account_id, date_str=date_str)
    except Exception as exc:  # pragma: no cover - 파이프라인 예외 방어
        logger.error("%s 추천 데이터 생성 중 오류가 발생했습니다: %s", account_id.upper(), exc)
        return []

    if not result:
        logger.warning("%s에 대한 추천을 생성하지 못했습니다.", account_id.upper())
        return []

    if isinstance(result, list):
        items = result
    else:
        recommendations = getattr(result, "recommendations", None)
        if not recommendations:
            logger.warning("%s 추천 결과가 비어 있습니다.", account_id.upper())
            return []

        header_line = getattr(result, "header_line", "")
        if header_line:
            header_plain = strip_html_tags(str(header_line)).strip()
            if header_plain:
                logger.info(header_plain)

        items = list(recommendations)

    return items


def dump_json(data: Any, path: Path) -> None:
    """Persist recommendation data to disk in a readable JSON format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, sort_keys=False, default=str)


def print_run_header(account_id: str, *, date_str: Optional[str]) -> None:
    banner = f"=== {account_id.upper()} 추천 생성 ==="
    logger.info("%s", banner)
    logger.info("기준일: %s", date_str or "auto (latest trading day)")


def print_result_summary(
    items: List[Dict[str, Any]], account_id: str, date_str: Optional[str] = None
) -> None:
    """Emit a condensed summary of recommendation results to stdout."""

    if not items:
        logger.warning("%s에 대한 결과가 없습니다.", account_id.upper())
        return

    state_counts = Counter(item.get("state", "UNKNOWN") for item in items)
    state_summary = ", ".join(f"{state}: {count}" for state, count in sorted(state_counts.items()))

    base_date = items[0].get("base_date") if items else (date_str or "N/A")

    logger.info("=== %s 추천 요약 (기준일: %s) ===", account_id.upper(), base_date)

    preview_count = min(10, len(items))
    if preview_count > 0:
        logger.info("상위 %d개 항목 미리보기:", preview_count)
        headers = ["순위", "티커", "종목명", "카테고리", "상태", "점수", "일간수익률", "보유일", "문구"]
        aligns = ["right", "left", "left", "left", "center", "right", "right", "right", "left"]
        rows: List[List[str]] = []

        for item in items[:preview_count]:
            holding_days = item.get("holding_days")
            holding_days_str = (
                f"{int(holding_days)}" if isinstance(holding_days, (int, float)) else "-"
            )

            rows.append(
                [
                    str(item.get("rank", "-")),
                    str(item.get("ticker", "-")),
                    str(item.get("name", "-")),
                    str(item.get("category", "-")),
                    str(item.get("state", "-")),
                    f"{item.get('score', 0):.2f}"
                    if isinstance(item.get("score"), (int, float))
                    else "-",
                    f"{item.get('daily_pct', 0):.2f}%"
                    if isinstance(item.get("daily_pct"), (int, float))
                    else "-",
                    holding_days_str,
                    str(item.get("phrase", "")),
                ]
            )

        for line in render_table_eaw(headers, rows, aligns):
            logger.info("%s", line)

    if state_summary:
        logger.info("상태 요약: %s", state_summary)
    buy_count = sum(1 for item in items if item.get("state") == "BUY")
    logger.info("매수 추천: %d개, 대기: %d개", buy_count, len(items) - buy_count)
    logger.info("결과가 성공적으로 생성되었습니다. (총 %d개 항목)", len(items))


invoke_country_pipeline = invoke_account_pipeline  # 하위 호환


__all__ = [
    "dump_json",
    "invoke_account_pipeline",
    "invoke_country_pipeline",
    "print_result_summary",
    "print_run_header",
]
