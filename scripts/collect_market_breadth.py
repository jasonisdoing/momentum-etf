"""시장 폭(ADR) 집계 배치.

코스피 200 / 코스닥 150(시가총액 상위)의 일별 상승·하락 종목수를 `market_breadth_daily`
에 적립한다. `/market-trend` 의 ADR 차트가 이 값을 읽는다.

화면 접근과 무관하게 돌아야 하므로 대상 종목 선정도 이 배치가 직접 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present  # noqa: E402
from utils.market_breadth_service import refresh_market_breadth  # noqa: E402


def main() -> int:
    load_env_if_present()
    summary = refresh_market_breadth()

    parts = [
        f"{market}=대상{info['universe']}·신규{info['inserted']}일·갱신{info['updated']}일({info['latest_date']})"
        for market, info in summary["markets"].items()
    ]
    print(f"[market_breadth] 시장 폭 집계 완료: {' '.join(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
