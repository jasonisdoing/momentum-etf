"""한국 지수 구성종목을 갱신한다 (KOSPI200 · KOSDAQ150).

한국은 공식 구성종목 API 가 없어 **추종 ETF 의 보유종목**을 명단으로 쓴다.
어떤 ETF 를 볼지는 `index_constituents_loader.KOR_INDEX_SOURCES` 가 단일 소스다.

  KOSPI200  ← KODEX 200(069500)
  KOSDAQ150 ← KODEX 코스닥150(229200)

`/kor-market-stock` 의 지수 토글이 이 명단을 읽고, `/kor-dividend` 는 저장된 KOSPI200 을
유니버스로 쓴다. 미국·호주의 `update_us_market_stocks.py` / `update_aus_market_stocks.py`
와 같은 자리다.

구성종목 수가 기대 범위를 벗어나면 저장하지 않고 실패로 끝난다(종료 코드 1).
원본 구조가 바뀌었을 때 조용히 낡은 명단을 계속 쓰는 상황을 막기 위한 것이다.

사용법:
    python scripts/update_kor_market_stocks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.index_constituents_loader import (  # noqa: E402
    KOR_INDEX_SOURCES,
    refresh_kor_index_from_etf,
)
from utils.logger import get_app_logger  # noqa: E402


def main() -> None:
    logger = get_app_logger()
    total = len(KOR_INDEX_SOURCES)
    for step, (index, source) in enumerate(KOR_INDEX_SOURCES.items(), start=1):
        print(f"[{step}/{total}] {index} 구성종목 갱신 ({source['etf_name']}({source['etf_ticker']}) 보유종목)...")
        result = refresh_kor_index_from_etf(index)
        logger.info(
            "[KOR INDEX] %s 구성종목 %d개 저장 (기준일 %s)",
            result["index"],
            result["count"],
            result["as_of_date"] or "-",
        )
        print(f"  저장 완료: {result['count']}개 (ETF 기준일 {result['as_of_date'] or '-'})")


if __name__ == "__main__":
    main()
