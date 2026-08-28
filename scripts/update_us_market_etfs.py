"""미국 ETF 마켓 목록 캐시를 갱신한다 (/us-market-etf 화면).

KIS 미국 3개 거래소 마스터에서 ETF 유니버스를 만들고, yfinance 일봉으로
거래대금 상위 N 개의 수익률(일간·1/2/3개월)을 계산해 DB 에 저장한다.
로직은 전부 `utils/us_etf_market_service.py` 에 있다.

사용법:
    python scripts/update_us_market_etfs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.env import load_env_if_present  # noqa: E402

load_env_if_present()

from utils.us_etf_market_service import refresh_us_etf_market_cache  # noqa: E402


def main() -> None:
    try:
        count = refresh_us_etf_market_cache()
    except Exception as exc:
        # 종료 코드를 남겨야 cron 래퍼가 실패로 보고 슬랙 알림을 보낸다.
        print(f"미국 ETF 마켓 캐시 갱신 실패: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"미국 ETF 마켓 캐시 저장: {count}건")


if __name__ == "__main__":
    main()
