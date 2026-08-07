"""네이버 차트(fchart, legacy XML) 일봉 조회 공통 헬퍼.

지수(KOSPI, KPI200)와 개별 종목/ETF(티커) 모두 같은 API 를 쓴다.
count 만큼 최신부터 거꾸로 N건 반환.
"""

from __future__ import annotations

import re

import pandas as pd
import requests

from config import NAVER_FINANCE_CHART_API_URL
from utils.logger import get_app_logger

logger = get_app_logger()

_NAVER_ITEM_RE = re.compile(r'<item data="([^"]+)"')


def fetch_naver_daily_ohlc(symbol: str, count: int) -> pd.DataFrame | None:
    """네이버 차트 API 에서 일봉 OHLCV 시계열을 받아온다.

    Args:
        symbol: 지수 심볼(KOSPI, KPI200 등) 또는 6자리 티커.
        count: 최근부터 N 거래일.

    Returns:
        DatetimeIndex 오름차순 정렬된 pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
        또는 None(조회 실패/데이터 없음).
    """
    try:
        resp = requests.get(
            NAVER_FINANCE_CHART_API_URL,
            params={"symbol": symbol, "timeframe": "day", "count": int(count), "requestType": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.encoding = "EUC-KR"
        items = _NAVER_ITEM_RE.findall(resp.text)
    except Exception:
        logger.exception("네이버 차트 조회 실패: %s", symbol)
        return None
    if not items:
        return None

    dates: list[pd.Timestamp] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    for raw in items:
        parts = raw.split("|")
        if len(parts) < 5:
            continue
        try:
            ts = pd.Timestamp(parts[0])
            open_val = float(parts[1])
            high_val = float(parts[2])
            low_val = float(parts[3])
            close_val = float(parts[4])
            volume_val = float(parts[5]) if len(parts) >= 6 else 0.0
        except (ValueError, TypeError):
            continue
        dates.append(ts)
        opens.append(open_val)
        highs.append(high_val)
        lows.append(low_val)
        closes.append(close_val)
        volumes.append(volume_val)

    if not dates:
        return None
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates),
    )
    return df.sort_index()
