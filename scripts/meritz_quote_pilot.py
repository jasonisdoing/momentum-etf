"""대표 종목의 메리츠 시세를 읽어 데이터 소스 화면에 검증 결과를 남긴다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.meritz_quote_pilot import MARKETS, collect_quote_pilot
from utils.env import load_env_if_present


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True, help="메리츠 API가 연동된 계좌 ID")
    parser.add_argument("--country", choices=MARKETS, required=True)
    parser.add_argument("--seconds", type=int, default=35, help="웹소켓 관찰 시간 (5~120초)")
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120:
        parser.error("--seconds는 5~120 범위여야 합니다.")
    load_env_if_present()
    report = collect_quote_pilot(args.account, args.country, args.seconds)
    verified = [ticker for ticker, quote in report["websocket"].items() if quote["verified"]]
    print(f"{args.country}/{report['session']} 체결 확인: {', '.join(verified) or '없음'}")
    print(f"REST {len(report['rest'])}종 · 기존 소스 {len(report['baseline'])}종 · {report['checked_at']}")
    if report["issues"]:
        print("수집 참고: " + " / ".join(report["issues"]))
    return 0 if len(verified) == len(report["targets"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
