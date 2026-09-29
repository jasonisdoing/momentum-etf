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

마지막에 `/kor-market-stock` 「통합」 보기와 같은 명단(코스피·코스닥 시총 상위, 개별주)
중 종목풀에 없는 종목을 슬랙으로 알린다 — 미국 배치와 같은 기준이다. 지수 명단(ETF
보유종목)이 아니라 화면에 보이는 종목이 대상이다.

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
from utils.index_pool_alert import notify_unregistered_market_stocks  # noqa: E402
from utils.kor_stock_market_service import KOR_MARKET_TOP_COUNTS, kor_market_top_tickers  # noqa: E402
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

    # 미등록 알림 — 화면 「통합」 보기와 같은 명단(지수 명단이 아니다).
    alert_items = {
        market: [{"ticker": ticker} for ticker in kor_market_top_tickers(market)] for market in KOR_MARKET_TOP_COUNTS
    }
    count = notify_unregistered_market_stocks("kor", alert_items)
    print(f"한국 시장 통합 종목 종목풀 미등록: {count}개")


if __name__ == "__main__":
    main()
