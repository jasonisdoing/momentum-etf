"""Helpers for CLI recommendation output."""

from __future__ import annotations

import json
import numbers
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.notification import strip_html_tags
from utils.report import render_table_eaw
from utils.logger import get_app_logger
from utils.formatters import format_price_deviation, format_price

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


def print_result_summary(items: List[Dict[str, Any]], account_id: str, date_str: Optional[str] = None) -> None:
    """Emit a condensed summary of recommendation results to stdout."""

    if not items:
        logger.warning("%s에 대한 결과가 없습니다.", account_id.upper())
        return

    state_counts = Counter(item.get("state", "UNKNOWN") for item in items)
    state_summary = ", ".join(f"{state}: {count}" for state, count in sorted(state_counts.items()))

    base_date = items[0].get("base_date") if items else (date_str or "N/A")

    logger.info("=== %s 추천 요약 (기준일: %s) ===", account_id.upper(), base_date)

    # 보유 종목(HOLD, SELL_*)은 항상 포함, 나머지는 상위 10개
    holding_states = {"HOLD", "HOLD_CORE", "SELL_TREND", "SELL_RSI", "SELL_REPLACE", "CUT_STOPLOSS"}
    held_items = [item for item in items if item.get("state") in holding_states]
    other_items = [item for item in items if item.get("state") not in holding_states]

    preview_items = held_items + other_items[: max(0, 10 - len(held_items))]

    if preview_items:
        logger.info("상위 %d개 항목 미리보기 (보유 %d개 포함):", len(preview_items), len(held_items))
        headers = [
            "순위",
            "티커",
            "종목명",
            "카테고리",
            "상태",
            "점수",
            "RSI",
            "일간수익률",
            "보유일",
            "문구",
        ]
        aligns = ["right", "left", "left", "left", "center", "right", "right", "right", "right", "left"]
        rows: List[List[str]] = []

        for item in preview_items:
            holding_days = item.get("holding_days")
            holding_days_str = f"{int(holding_days)}" if isinstance(holding_days, (int, float)) else "-"

            rows.append(
                [
                    str(item.get("rank", "-")),
                    str(item.get("ticker", "-")),
                    str(item.get("name", "-")),
                    str(item.get("category", "-")),
                    str(item.get("state", "-")),
                    (f"{item.get('score', 0):.2f}" if isinstance(item.get("score"), (int, float)) else "-"),
                    (f"{item.get('rsi_score', 0):.2f}" if isinstance(item.get("rsi_score"), (int, float)) else "-"),
                    (f"{item.get('daily_pct', 0):.2f}%" if isinstance(item.get("daily_pct"), (int, float)) else "-"),
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


def dump_recommendation_log(
    report: Any,
    *,
    results_dir: Optional[Path | str] = None,
) -> Path:
    """추천 결과를 로그 파일로 저장합니다."""
    from pathlib import Path

    account_id = getattr(report, "account_id", "unknown")
    base_date = getattr(report, "base_date", None)
    recommendations = getattr(report, "recommendations", [])

    # 기본 디렉토리 설정 (계정별 폴더)
    if results_dir is None:
        # 프로젝트 루트의 zresults/<account> 디렉토리
        base_dir = Path(__file__).parent.parent.parent / "zresults" / account_id
    else:
        base_dir = Path(results_dir) / account_id

    base_dir.mkdir(parents=True, exist_ok=True)

    # 파일명 생성 (실제 실행 날짜 사용)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = base_dir / f"recommend_{date_str}.log"

    lines: List[str] = []

    # 헤더
    lines.append(f"추천 로그 생성: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    base_date_str = base_date.strftime("%Y-%m-%d") if hasattr(base_date, "strftime") else str(base_date)
    lines.append(f"계정: {account_id.upper()} | 기준일: {base_date_str}")
    lines.append("")

    # 상태별 카운트
    state_counts = Counter(item.get("state", "UNKNOWN") for item in recommendations)
    lines.append("=== 상태 요약 ===")
    for state, count in sorted(state_counts.items()):
        lines.append(f"  {state}: {count}개")
    lines.append("")

    # 추천 목록 (테이블 형식)
    lines.append("=== 추천 목록 ===")
    lines.append("")

    # 테이블 헤더 (화면 UI와 동일)
    country_code = getattr(report, "country_code", "")
    country_lower = (country_code or "").strip().lower()
    nav_mode = country_lower in {"kr", "kor"}
    show_deviation = country_lower in {"kr", "kor"}

    price_header = "현재가"

    headers = [
        "#",
        "티커",
        "종목명",
        "카테고리",
        "상태",
        "보유일",
        "일간(%)",
        "평가(%)",
        price_header,
    ]
    if nav_mode:
        headers.append("Nav")
    if show_deviation:
        headers.append("괴리율")
    headers.extend(["점수", "RSI", "지속", "문구"])

    aligns = [
        "right",
        "left",
        "left",
        "left",
        "center",
        "right",
        "right",
        "right",
        "right",
    ]
    if nav_mode:
        aligns.append("right")
    if show_deviation:
        aligns.append("right")
    aligns.extend(["right", "right", "right", "left"])

    # 테이블 데이터
    rows: List[List[str]] = []
    for item in recommendations:
        rank = item.get("rank", 0)
        ticker = item.get("ticker", "-")
        name = item.get("name", "-")
        category = item.get("category", "-")
        state = item.get("state", "-")
        holding_days = item.get("holding_days", 0)
        daily_pct = item.get("daily_pct", 0)
        evaluation_pct = item.get("evaluation_pct", 0)
        price = item.get("price")
        nav_price = item.get("nav_price")
        price_deviation = item.get("price_deviation")
        score = item.get("score", 0)
        rsi_score = item.get("rsi_score", 0)
        streak = item.get("streak", 0)
        phrase = item.get("phrase", "")

        row = [
            str(rank),
            ticker,
            name,
            category,
            state,
            str(holding_days) if holding_days > 0 else "-",
            f"{daily_pct:+.2f}%" if isinstance(daily_pct, (int, float)) else "-",
            f"{evaluation_pct:+.2f}%" if isinstance(evaluation_pct, (int, float)) and evaluation_pct != 0 else "-",
            format_price(price, country_code),
        ]
        if nav_mode:
            row.append(format_price(nav_price, country_code))
        if show_deviation:
            row.append(format_price_deviation(price_deviation))
        row.extend(
            [
                f"{score:.1f}" if isinstance(score, (int, float)) else "-",
                f"{rsi_score:.1f}" if isinstance(rsi_score, (int, float)) else "-",
                f"{streak}일" if streak > 0 else "-",
                phrase,
            ]
        )
        rows.append(row)

    # 테이블 렌더링
    table_lines = render_table_eaw(headers, rows, aligns)
    lines.extend(table_lines)
    lines.append("")

    # 파일 저장
    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    return path


invoke_country_pipeline = invoke_account_pipeline  # 하위 호환


__all__ = [
    "dump_json",
    "dump_recommendation_log",
    "invoke_account_pipeline",
    "invoke_country_pipeline",
    "print_result_summary",
    "print_run_header",
]
