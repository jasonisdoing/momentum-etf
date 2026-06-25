"""백테스트 실행 엔트리 포인트 (스위칭 전략). 시장은 config 의 'market' 필드로 결정된다."""

import sys
from datetime import datetime

from leverage.constants import CONFIG_DIR, INITIAL_CAPITAL_KRW, ZRESULTS_DIR
from leverage.engine.backtest.runner import run_backtest
from leverage.engine.backtest.settings import load_settings


def main() -> None:
    # CLI 인자로 전략 프로파일 지정 (기본값: switch)
    profile = sys.argv[1] if len(sys.argv) > 1 else "switch"
    config_path = CONFIG_DIR / f"{profile}.json"

    if not config_path.exists():
        print(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return

    settings = load_settings(config_path)
    try:
        _run_switch(profile, settings)
    except Exception as exc:
        if "YFRateLimitError" in repr(exc) or "rate limit" in repr(exc).lower():
            print("YFRateLimitError: 요청이 너무 많습니다. 잠시 후 다시 실행하세요.")
            return
        raise


def _run_switch(profile: str, settings: dict) -> None:
    report = run_backtest(settings)

    out_dir = ZRESULTS_DIR / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{datetime.now().date()}.log"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"백테스트 로그 생성: {datetime.now().isoformat()}\n")
        f.write(f"프로파일: {profile} | 초기자본: {INITIAL_CAPITAL_KRW:,}\n")
        f.write(f"시작일: {report['start']} | 종료일: {report['end']}\n\n")
        f.write("2. ========= 일자별 성과 ==========\n\n")
        if report.get("segment_lines"):
            f.write("=== 구간별 보유 요약 ===\n")
            for line in report["segment_lines"]:
                f.write(line + "\n")
            f.write("\n=== 일자별 상세 ===\n")
        for line in report["daily_log"]:
            f.write(line + "\n")
        f.write("\n")
        for line in report.get("used_settings_lines", []):
            f.write(line + "\n")
        f.write("\n")
        if report.get("weekly_summary_lines"):
            for line in report["weekly_summary_lines"]:
                f.write(line + "\n")
            f.write("\n")
        if report.get("monthly_summary_lines"):
            for line in report["monthly_summary_lines"]:
                f.write(line + "\n")
            f.write("\n")
        for line in report["asset_summary_lines"]:
            f.write(line + "\n")
        f.write("\n")
        for line in report["summary_lines"]:
            f.write(line + "\n")
        if report.get("bench_table_lines"):
            for line in report["bench_table_lines"]:
                f.write(line + "\n")

    print(f"=== Backtest 결과 ({profile}) ===")
    for k, v in report.items():
        if k == "daily_log" or k.endswith("_lines"):
            continue
        print(f"{k}: {v}")
    if report.get("used_settings_lines"):
        print("\n".join(report["used_settings_lines"]))
    print("\n".join(report["asset_summary_lines"]))
    print("\n".join(report["summary_lines"]))
    if report.get("bench_table_lines"):
        for line in report["bench_table_lines"]:
            print(line)

    print(f"백테스트 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
