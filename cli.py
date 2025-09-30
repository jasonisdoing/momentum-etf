"""MomentumEtf 프로젝트용 국가 기반 CLI."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.account_registry import (
    CountrySettingsError,
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
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="JSON 출력 시 보기 좋게 정렬",
    )
    return parser


def _invoke_country_pipeline(country: str, *, date_str: str | None) -> Any:
    pipeline = importlib.import_module("logic.signals.pipeline")
    handler = getattr(pipeline, "generate_country_signal_report", None)
    if handler is None:
        raise NotImplementedError(
            "logic.signals.pipeline.generate_country_signal_report(country=..., date_str=...) 함수를 구현하세요."
        )
    return handler(country=country, date_str=date_str)


def _dump_json(data: Any, path: Path, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        if pretty:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        else:
            json.dump(data, fp, ensure_ascii=False)


def _print_run_header(country: str, *, date_str: str | None) -> None:
    banner = f"=== {country.upper()} 시그널 생성 ==="
    print("\n" + banner)
    print(f"[INFO] 기준일: {date_str or 'auto (latest trading day)'}")


def _print_result_summary(country: str, items: Iterable[Mapping[str, Any]]) -> None:
    items_list = list(items)
    if not items_list:
        print(f"[WARN] {country.upper()} 결과가 비어 있습니다.")
        return

    state_counts = Counter(str(item.get("state", "-")) for item in items_list)
    state_summary = ", ".join(f"{state}: {count}" for state, count in sorted(state_counts.items()))
    print(f"[INFO] 총 {len(items_list)}개 항목 ({state_summary})")

    preview_count = 10
    print(f"[INFO] 상위 {preview_count}개 미리보기:")
    headers = ["티커", "종목명", "카테고리", "상태", "점수"]
    aligns = ["left", "left", "left", "center", "right"]
    rows = []
    for row in items_list[:preview_count]:
        ticker = str(row.get("ticker", "-"))
        name = str(row.get("name", "-"))
        category = str(row.get("category", "-"))
        state = str(row.get("state", "-"))
        score = row.get("score")
        score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        rows.append([ticker, name, category, state, score_display])

    for line in render_table_eaw(headers, rows, aligns):
        print(line)

    if len(items_list) > preview_count:
        print(f"[INFO] ... (총 {len(items_list)}개 중 {preview_count}개 표시)")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    country = args.country.lower()

    # 설정 로드로 국가 코드 유효성 추가 확인
    try:
        get_country_settings(country)
        get_strategy_rules(country)
    except CountrySettingsError as exc:
        parser.error(str(exc))

    _print_run_header(country, date_str=args.date)

    try:
        result = _invoke_country_pipeline(country, date_str=args.date)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from exc

    items = result if isinstance(result, list) else []
    _print_result_summary(country, items)

    output_path = Path(args.output) if args.output else RESULTS_DIR / f"{country}.json"

    try:
        _dump_json(result, output_path, pretty=bool(args.pretty))
    except TypeError as exc:  # JSON 직렬화 실패 시 디버그 도움
        raise SystemExit(f"JSON 직렬화에 실패했습니다: {exc}") from exc

    print(f"✅ {country.upper()} 결과를 '{output_path}'에 저장했습니다.")


if __name__ == "__main__":
    main()
