"""한국 배당주 화면(`/kor-dividend`) 데이터를 갱신한다.

종목별 지표(배당률·주주환원율·추세)를 DART + 네이버에서 모아 적재한다.

유니버스(KOSPI200 구성종목)는 **여기서 만들지 않는다** — `update_kor_market_stocks.py`
가 적재한 것을 읽기만 한다. 그 배치가 먼저 돌아야 한다(crontab 순서도 그렇게 잡혀 있다).

실패하면 예외를 그대로 올린다. 배치 러너(`infra/cron/run_batch.py`)가 종료 코드와
마지막 로그를 슬랙으로 보낸다 — 여기서 잡아 삼키면 지표가 조용히 낡는다.

사용법:
    python scripts/update_kor_dividend_stocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.kor_dividend_service import refresh_kor_dividend_stocks  # noqa: E402


def main() -> None:
    print("종목별 재무·배당 지표 수집 (DART + 네이버)...")
    metrics = refresh_kor_dividend_stocks()
    years = metrics["years"]
    print(
        f"  적재 완료: {metrics['saved']}종목 · 회계연도 {years[0]}~{years[-1]}"
        f" · 유니버스 이탈 제거 {metrics['removed']}"
        f" · 결측 있음 {metrics['with_notes']}"
    )


if __name__ == "__main__":
    main()
