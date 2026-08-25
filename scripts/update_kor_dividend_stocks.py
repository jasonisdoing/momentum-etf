"""한국 배당주 화면(`/kor-dividend`) 데이터를 갱신한다.

  [1] 유니버스 — KODEX 200(069500) 보유종목을 KOSPI200 구성종목으로 저장
  [2] 종목별 지표 — (다음 단계에서 추가) 배당률·주주환원율·추세·점수

실패하면 예외를 그대로 올린다. 배치 러너(`infra/cron/run_batch.py`)가 종료 코드와
마지막 로그를 슬랙으로 보낸다 — 여기서 잡아 삼키면 유니버스가 조용히 낡는다.

사용법:
    python scripts/update_kor_dividend_stocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.kor_dividend_service import (  # noqa: E402
    refresh_kor_dividend_stocks,
    refresh_kospi200_constituents,
)


def main() -> None:
    print("[1/2] KOSPI200 구성종목 갱신 (KODEX 200 보유종목)...")
    universe = refresh_kospi200_constituents()
    print(f"  저장 완료: {universe['count']}개 (ETF 기준일 {universe['as_of_date'] or '-'})")

    print("[2/2] 종목별 재무·배당 지표 수집 (DART + 네이버)...")
    metrics = refresh_kor_dividend_stocks()
    years = metrics["years"]
    print(
        f"  적재 완료: {metrics['saved']}종목 · 회계연도 {years[0]}~{years[-1]}"
        f" · 유니버스 이탈 제거 {metrics['removed']}"
        f" · 결측 있음 {metrics['with_notes']}"
    )


if __name__ == "__main__":
    main()
