"""프로젝트 전역에서 사용하는 설정 값 정의."""

from datetime import time

import numpy as np

CACHE_START_DATE = "2024-01-01"
# 포트폴리오 카테고리별 최대 보유 수
MAX_PER_CATEGORY = 1

# 네이버 금융 API 설정 (비공식 API)
NAVER_FINANCE_ETF_API_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
NAVER_FINANCE_CHART_API_URL = "https://fchart.stock.naver.com/sise.nhn"
NAVER_FINANCE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://finance.naver.com/sise/etfList.nhn",
    "Accept": "application/json, text/plain, */*",
}

MARKET_REGIME_FILTER_TICKER_MAIN = "^GSPC"
MARKET_REGIME_FILTER_TICKERS_AUX = ["^IXIC", "NQ=F", "^DJI"]

MARKET_REGIME_FILTER_COUNTRY = "us"
MARKET_REGIME_FILTER_MA_PERIOD = 50

# RSI 계산 파라미터 (원본 RSI 사용: 70 이상 과매수, 30 이하 과매도)
RSI_CALCULATION_CONFIG = {
    "period": 15,
    "ema_smoothing": 2.0,
}


# 통합 시장 거래 시간표
MARKET_SCHEDULES = {
    "kor": {
        "open": time(9, 10),
        "close": time(15, 50),
        "interval_minutes": 10,
        "timezone": "Asia/Seoul",
    },
    "aus": {
        "open": time(8, 20),
        "close": time(14, 20),
        "interval_minutes": 10,
        "timezone": "Asia/Seoul",
    },
}

# 백테스트 체결 슬리피지 가정치 (%)
BACKTEST_SLIPPAGE = {
    "kor": {
        "buy_pct": 0.25,
        "sell_pct": 0.25,
    },
    "aus": {
        "buy_pct": 0.5,
        "sell_pct": 0.5,
    },
    "us": {
        "buy_pct": 0.15,
        "sell_pct": 0.15,
    },
}

# 실시간 ETF 시세 소스 선택, 허용 값: "Price", "Nav"
KOR_REALTIME_ETF_PRICE_SOURCE = "Price"
# KOR_REALTIME_ETF_PRICE_SOURCE = "Nav"

# 튜닝·최적화 작업이 공유하는 계정별 파라미터 탐색 설정
ACCOUNT_PARAMETER_SEARCH_CONFIG: dict[str, dict] = {
    "a1": {
        "MA_RANGE": np.arange(30, 65, 5),
        "MA_TYPE": ["SMA"],
        "PORTFOLIO_TOPN": [7],
        "REPLACE_SCORE_THRESHOLD": np.arange(2, 8, 1),
        "OVERBOUGHT_SELL_THRESHOLD": [100],
        "CORE_HOLDINGS": ["ASX:HNDQ", "ASX:IOO", "ASX:FANG"],
        "COOLDOWN_DAYS": [1, 2],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
    "k1": {
        "MA_RANGE": np.arange(80, 105, 5),
        "MA_TYPE": ["HMA"],
        "PORTFOLIO_TOPN": [10],
        "REPLACE_SCORE_THRESHOLD": np.arange(2, 8, 1),
        "OVERBOUGHT_SELL_THRESHOLD": np.arange(90, 101, 1),
        "CORE_HOLDINGS": [],
        "COOLDOWN_DAYS": [1, 2],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
}
