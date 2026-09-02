"""시장 폭(ADR) 집계 배치.

두 가지를 한 컬렉션(`market_breadth_daily`)에 적립한다.

  · **시장 4개** — 코스피 200 / 코스닥 150(시가총액 상위) · S&P 500 / 나스닥 100(구성종목).
  · **종목풀** — 활성 종목풀 각각의 폭. 지수와 구성이 달라(us_stock 은 S&P100+나스닥100
    조합) 매매하는 종목의 폭을 그대로 본다. 가격 캐시만 읽어 외부 조회가 없다.

`/market-trend` 의 ADR 차트와 전략의 ADR 하한 게이트가 이 값을 읽는다.
화면 접근과 무관하게 돌아야 하므로 대상 종목 선정도 이 배치가 직접 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present  # noqa: E402
from utils.market_breadth_service import refresh_market_breadth, refresh_pool_breadth  # noqa: E402


def main() -> int:
    load_env_if_present()
    summary = refresh_market_breadth()

    # 스킵된 시장(기준일 불일치 등)은 요약 형태가 다르다 — 정상 형태만 가정하면 KeyError 로 죽는다.
    parts = [
        f"{market}=스킵({info['reason']})"
        if info.get("market_skipped")
        else f"{market}=대상{info['universe']}·신규{info['inserted']}일·갱신{info['updated']}일({info['latest_date']})"
        for market, info in summary["markets"].items()
    ]
    print(f"[market_breadth] 시장 폭 집계 완료: {' '.join(parts)}")

    # 종목풀 ADR — 가격 캐시(2번 배치, 매시 20분)만 읽으므로 이 시점에 항상 최신이다.
    # 시장 집계가 실패해도 여기까지 오면 돌리고, 반대로 여기서 실패해도 위 결과는 남긴다.
    pool_summary = refresh_pool_breadth()
    pool_parts = [
        f"{pool}=실패({info['error']})"
        if info.get("error")
        else f"{pool}=스킵({info['reason']})"
        if info.get("skipped")
        else f"{pool}=대상{info['universe_size']}·신규{info['inserted']}일·갱신{info['updated']}일({info['latest_date']})"
        for pool, info in pool_summary["pools"].items()
    ]
    print(f"[market_breadth] 종목풀 ADR 집계 완료: {' '.join(pool_parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
