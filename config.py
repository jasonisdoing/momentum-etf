"""프로젝트 전역에서 사용하는 설정 값 정의."""

from datetime import time

CACHE_START_DATE = "2024-01-02"
# 포트폴리오 카테고리별 최대 보유 수
MAX_PER_CATEGORY = 1

# 카테고리 중복 제한에서 예외되는 카테고리 목록
# 이 카테고리들은 여러 종목을 동시에 보유할 수 있음
CATEGORY_EXCEPTIONS = ["기본"]

# 네이버 금융 API 설정
NAVER_FINANCE_ETF_API_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
NAVER_FINANCE_CHART_API_URL = "https://fchart.stock.naver.com/sise.nhn"
NAVER_FINANCE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://finance.naver.com/sise/etfList.nhn",
    "Accept": "application/json, text/plain, */*",
}

# 호주 MarketIndex QuoteAPI 설정
AU_QUOTEAPI_URL = "https://quoteapi.com/api/v5/symbols"
AU_QUOTEAPI_APP_ID = "af5f4d73c1a54a33"  # marketindex.com.au 제공
AU_QUOTEAPI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.marketindex.com.au/",
    "Origin": "https://www.marketindex.com.au",
}

# RSI 계산 파라미터 (원본 RSI 사용: 70 이상 과매수, 30 이하 과매도)
RSI_CALCULATION_CONFIG = {
    "period": 15,
    "ema_smoothing": 2.0,
}


# 통합 시장 거래 시간표
MARKET_SCHEDULES = {
    "kor": {
        "open": time(9, 0),
        "close": time(16, 0),
        "open_offset_minutes": 30,
        "close_offset_minutes": 30,
        "timezone": "Asia/Seoul",
    },
    "us": {
        "open": time(9, 30),
        "close": time(16, 0),
        "open_offset_minutes": 30,
        "close_offset_minutes": 30,
        "timezone": "America/New_York",
    },
    "au": {
        "open": time(10, 0),
        "close": time(16, 0),
        "open_offset_minutes": 30,
        "close_offset_minutes": 30,
        "timezone": "Australia/Sydney",
    },
}

# 백테스트 체결 슬리피지 가정치 (%)
BACKTEST_SLIPPAGE = {
    "kor": {
        "buy_pct": 0.5,
        "sell_pct": 0.5,
    },
    "us": {
        "buy_pct": 0.25,
        "sell_pct": 0.25,
    },
    "au": {
        "buy_pct": 1.0,
        "sell_pct": 1.0,
    },
}

# 튜닝 앙상블 크기 (Top N)
# 상위 N개의 결과를 사용하여 파라미터를 결정합니다. (MA: 평균, 나머지: 최빈값)
# 반드시 홀수로 설정해야 합니다. (짝수 입력 시 런타임 에러 발생)
# Top1은 과거 데이터에 과최적화되므로 실전 성능이 불안정함.
# Top5 평균·최빈값 방식은 백테스트 성능이 약간 낮아져도 실전에서는 훨씬 안정적임.
# 실전용 전략에서는 TUNING_ENSEMBLE_SIZE = 5를 유지해야 장기 성과가 일정하게 나오므로 변경 금지.
TUNING_ENSEMBLE_SIZE = 1

# [전략 실행 시 실시간 가격 반영 여부]
# True: 장중 실행 시 현재가(실시간)를 '오늘 종가'로 가정하고 전략 실행 (순위 변동 발생 가능)
# False: 장중 실행 시 '어제 종가'까지만 전략에 반영 (순위 고정), 수익률만 실시간 업데이트 (권장)
USE_REALTIME_RECOMMENDATION = True
