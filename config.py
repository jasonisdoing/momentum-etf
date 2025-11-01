"""프로젝트 전역에서 사용하는 설정 값 정의."""

from datetime import time

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
        "open": time(9, 00),
        "close": time(16, 00),
        "interval_minutes": 20,
        "timezone": "Asia/Seoul",
    },
    "aus": {
        "open": time(8, 00),
        "close": time(15, 00),
        "interval_minutes": 20,
        "timezone": "Asia/Seoul",
    },
}

# 백테스트 체결 슬리피지 가정치 (%)
BACKTEST_SLIPPAGE = {
    "kor": {
        "buy_pct": 0.5,
        "sell_pct": 0.5,
    },
    "aus": {
        "buy_pct": 1.0,
        "sell_pct": 1.0,
    },
    "us": {
        "buy_pct": 0.3,
        "sell_pct": 0.3,
    },
}

# 시장 지수 기반 타이밍에 사용할 지수 티커 (정보성 데이터만)
MARKET_TIMING_INDEX = {
    "kor": "^KS11",  # 코스피 200 ETF
    "aus": "^AXJO",  # ASX 200
}
# 한국 종목은 pykrx + 네이버 비공식 API
# 호주 종목은 yfinance
