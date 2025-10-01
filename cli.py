"""MomentumEtf 프로젝트용 국가 기반 CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Dict, Optional
import pandas as pd
from utils.account_registry import (
    get_country_settings,
    get_strategy_rules,
    list_available_countries,
)
from utils.report import render_table_eaw


RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _available_country_choices() -> list[str]:
    choices = list_available_countries()
    if not choices:
        raise SystemExit("국가 설정(JSON)이 존재하지 않습니다. data/settings/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 국가 시그널 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "country",
        choices=_available_country_choices(),
        help="실행할 국가 코드",
    )
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD 형식의 기준일 (미지정 시 파이프라인 기본값 사용)",
    )
    parser.add_argument(
        "--output",
        help="결과 JSON을 저장할 경로 (기본: data/results/{country}.json)",
    )
    # JSON은 항상 보기 좋게 정렬되어 출력됩니다.
    return parser


def _invoke_country_pipeline(country: str, *, date_str: str | None) -> List[Dict[str, Any]]:
    """국가별 시그널을 생성하고 결과를 반환합니다.

    Args:
        country: 국가 코드 (예: 'kor', 'aus')
        date_str: 기준일 (YYYY-MM-DD 형식), None인 경우 오늘 날짜 사용

    Returns:
        List[Dict[str, Any]]: 시그널 결과 리스트
    """
    from logic.signals import generate_signal_report

    signals = generate_signal_report(
        country=country,
        date_str=date_str,
    )

    if not signals:
        print(f"[WARN] {country.upper()}에 대한 시그널을 생성하지 못했습니다.")
        return []

    # print(f"\n[DEBUG] Generated signals for {country}:")
    # print(f"- Signals count: {len(signals)}")
    # print("\nSample signal data (first 3 items):")
    # for i, signal in enumerate(signals[:3], 1):
    #     print(f"{i}. {signal.get('ticker')} - {signal.get('name')}")
    #     print(f"   State: {signal.get('state')}, Daily %: {signal.get('daily_pct')}")
    #     print(f"   Price: {signal.get('price')}, Score: {signal.get('score')}")

    return signals


def _dump_json(data: Any, path: Path) -> None:
    """데이터를 JSON 파일로 저장합니다. 항상 보기 좋게 정렬된 형식으로 저장됩니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, sort_keys=False, default=str)


def _print_run_header(country: str, *, date_str: str | None) -> None:
    banner = f"=== {country.upper()} 시그널 생성 ==="
    print("\n" + banner)
    print(f"[INFO] 기준일: {date_str or 'auto (latest trading day)'}")


def _print_result_summary(
    items: List[Dict[str, Any]], country: str, date_str: str | None = None
) -> None:
    """시그널 결과를 요약하여 출력합니다.

    Args:
        items: _invoke_country_pipeline()의 결과 리스트
        country: 국가 코드
        date_str: 기준일 (선택 사항)
    """
    if not items:
        print(f"[WARN] {country.upper()}에 대한 결과가 없습니다.")
        return

    # 상태별 카운트
    from collections import Counter

    state_counts = Counter(item.get("state", "UNKNOWN") for item in items)
    state_summary = ", ".join(f"{state}: {count}" for state, count in sorted(state_counts.items()))

    # 기준일 설정
    base_date = items[0].get("base_date") if items else (date_str or "N/A")

    print(f"\n=== {country.upper()} 시그널 요약 (기준일: {base_date}) ===")
    print(f"[INFO] 총 {len(items)}개 항목 ({state_summary})")

    # 상위 10개 항목 미리보기
    preview_count = min(10, len(items))
    if preview_count > 0:
        print(f"\n[INFO] 상위 {preview_count}개 항목 미리보기:")
        headers = ["순위", "티커", "종목명", "카테고리", "상태", "점수", "일간수익률", "보유일"]
        aligns = ["right", "left", "left", "left", "center", "right", "right", "right"]
        rows = []

        for item in items[:preview_count]:
            holding_days = item.get("holding_days")
            if isinstance(holding_days, (int, float)):
                holding_days_str = f"{int(holding_days)}"
            else:
                holding_days_str = "-"

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
                ]
            )

        for line in render_table_eaw(headers, rows, aligns):
            print(line)

    # 추가 정보 출력
    buy_count = sum(1 for item in items if item.get("state") == "BUY")
    print(f"\n[INFO] 매수 추천: {buy_count}개, 대기: {len(items) - buy_count}개")
    print(f"[INFO] 결과가 성공적으로 생성되었습니다. (총 {len(items)}개 항목)")


def main() -> None:
    # 명령줄 인자 파싱
    parser = build_parser()
    args = parser.parse_args()

    # 국가 코드 유효성 검사
    country = args.country.lower()

    try:
        # 국가 설정 로드 (유효성 검사용)
        get_country_settings(country)
        get_strategy_rules(country)
    except Exception as exc:
        parser.error(f"국가 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    # 실행 헤더 출력
    _print_run_header(country, date_str=args.date)

    try:
        # 1. 시그널 생성
        items = _invoke_country_pipeline(country, date_str=args.date)

        # 2. 결과 출력
        _print_result_summary(items, country, args.date)

        # 3. 결과 저장
        output_path = Path(args.output) if args.output else RESULTS_DIR / f"{country}.json"
        _dump_json(items, output_path)

        print(f"\n✅ {country.upper()} 결과를 '{output_path}'에 저장했습니다.")

    except Exception as exc:
        raise SystemExit(f"오류가 발생했습니다: {exc}") from exc


if __name__ == "__main__":
    main()
