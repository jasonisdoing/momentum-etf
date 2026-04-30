"""프로젝트 전역에서 사용하는 설정 값 정의."""

import json
from pathlib import Path

CACHE_START_DATE = "2020-01-01"
SLACK_CHANNEL = "C0A0X2LTS3X"


# -----------------------------------------------------------------------
# 버킷(Bucket) 설정 및 스타일링
# -----------------------------------------------------------------------
_BUCKET_THEME_PATH = Path(__file__).resolve().parent / "shared" / "bucket_theme.json"
with _BUCKET_THEME_PATH.open("r", encoding="utf-8") as bucket_theme_file:
    _BUCKET_THEME = json.load(bucket_theme_file)

BUCKET_THEME = {int(bucket_id): value for bucket_id, value in (_BUCKET_THEME.get("buckets") or {}).items()}

BUCKET_CONFIG = {
    int(bucket_id): value
    for bucket_id, value in (_BUCKET_THEME.get("buckets") or {}).items()
    if int(bucket_id) in {1, 2, 3, 4}
}
CASH_CONFIG = BUCKET_THEME[5]

BUCKET_MAPPING = {k: v["name"] for k, v in BUCKET_CONFIG.items()}
ALL_BUCKET_MAPPING = {k: v["name"] for k, v in BUCKET_THEME.items()}
BUCKET_REVERSE_MAPPING = {v: k for k, v in BUCKET_MAPPING.items()}
BUCKET_OPTIONS = list(BUCKET_MAPPING.values())

# 네이버 금융 API 설정
NAVER_FINANCE_ETF_API_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
NAVER_FINANCE_CHART_API_URL = "https://fchart.stock.naver.com/sise.nhn"
NAVER_FINANCE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://finance.naver.com/sise/etfList.nhn",
    "Accept": "application/json, text/plain, */*",
}

# 네이버 ETF 카테고리/테마 API (투자국가/섹터/지수 등 ETF 분류 조회용)
NAVER_ETF_THEMES_URL = "https://stock.naver.com/api/stockSecurity/etfs/v1/domestic/themes"
NAVER_ETF_DOMESTIC_URL = "https://stock.naver.com/api/stockSecurity/etfs/v1/domestic"
NAVER_ETF_CATEGORY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://stock.naver.com/",
    "Accept": "application/json, text/plain, */*",
}

# 네이버 벌크 종목 시세 정보 (KOSPI/KOSDAQ)
NAVER_STOCK_MARKET_VALUE_URL = "https://m.stock.naver.com/api/stocks/marketValue/{market}"
NAVER_STOCK_MARKET_VALUE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/04.1",
    "Referer": "https://m.stock.naver.com/",
    "Accept": "application/json, text/plain, */*",
}
# 네이버 ETF 대분류 설정 (use: 대표 분류용, show: 개별 컬럼 표시용)
# 코드가 클수록 Representative(Main) 분류를 정할 때 우선순위가 높음
NAVER_ETF_CATEGORY_CONFIG = [
    {"code": "0101", "name": "주식", "use": True, "show": True},
    {"code": "0102", "name": "채권", "use": False, "show": False},
    {"code": "0103", "name": "부동산", "use": False, "show": False},
    {"code": "0104", "name": "멀티에셋", "use": False, "show": False},
    {"code": "0105", "name": "원자재", "use": False, "show": False},
    {"code": "0106", "name": "통화", "use": False, "show": False},
    {"code": "0108", "name": "단기자금(파킹형)", "use": False, "show": False},
    {"code": "0201", "name": "투자국가", "use": True, "show": True},
    {"code": "0301", "name": "배율", "use": False, "show": False},
    {"code": "0401", "name": "섹터", "use": True, "show": True},
    {"code": "0501", "name": "지수", "use": True, "show": True},
    {"code": "0601", "name": "혁신기술", "use": True, "show": True},
    {"code": "0606", "name": "투자전략", "use": False, "show": False},
    {"code": "0607", "name": "ESG", "use": False, "show": True},
    {"code": "0609", "name": "배당", "use": True, "show": True},
    {"code": "0610", "name": "단일종목", "use": False, "show": False},
    {"code": "0701", "name": "트렌드", "use": True, "show": True},
    {"code": "0803", "name": "국내운용사", "use": False, "show": False},
]

# 호주 MarketIndex QuoteAPI 설정
AU_QUOTEAPI_URL = "https://quoteapi.com/api/v5/symbols"
AU_QUOTEAPI_APP_ID = "af5f4d73c1a54a33"  # marketindex.com.au 제공
AU_QUOTEAPI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://www.marketindex.com.au/",
    "Origin": "https://www.marketindex.com.au",
}

# 토스증권 API 설정 (미국 주식 실시간)
TOSS_INVEST_API_BASE_URL = "https://wts-info-api.tossinvest.com"
TOSS_INVEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://tossinvest.com/",
    "Origin": "https://tossinvest.com",
    "Content-Type": "application/json",
}

# KIS 종목정보파일 다운로드 URL
KIS_KOSPI_MASTER_URL = "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip"
KIS_KOSDAQ_MASTER_URL = "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip"


# 통합 시장 거래 시간표
from datetime import time

MARKET_SCHEDULES = {
    "kor": {
        "open": time(9, 0),
        "close": time(15, 30),
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

# 1개월 = 20 거래일 (MA 개월 → 거래일 변환에 사용)
TRADING_DAYS_PER_MONTH = 20

# 지표 계산에 필요한 절대 최소 거래일 수 (MA 타입 무관, 항상 적용)
# ENABLE_DATA_SUFFICIENCY_CHECK = True  → MA 타입별 엄격 기준 적용 (60~120일)
# ENABLE_DATA_SUFFICIENCY_CHECK = False → 이 값만 체크 (신규 상장 ETF 조기 포착용)
# 5일(1주) 미만 데이터는 추세 판단이 불가하므로 제외
MIN_TRADING_DAYS = 5

# -----------------------------------------------------------------------
# 백테스트 파라미터 스윕 설정
# -----------------------------------------------------------------------
BACKTEST_START_DATE = "2025-05-01"
BACKTEST_INITIAL_KRW_AMOUNT = 100_000_000

# 슬리피지는 % 단위로 입력한다.
SLIPPAGE_CONFIG: dict[str, dict[str, float]] = {
    "kor_kr": {
        "BUY_PCT": 0.25,
        "SELL_PCT": 0.25,
    },
    "kor_us": {
        "BUY_PCT": 0.25,
        "SELL_PCT": 0.25,
    },
    "aus": {
        "BUY_PCT": 0.5,
        "SELL_PCT": 0.5,
    },
    "us": {
        "BUY_PCT": 0.15,
        "SELL_PCT": 0.15,
    },
    "kor": {
        "BUY_PCT": 0.25,
        "SELL_PCT": 0.25,
    },
}

BACKTEST_CONFIG: dict[str, dict] = {
    "kor_kr": {
        "BENCHMARK": {"ticker": "069500", "name": "KODEX 200"},
        "TOP_N_HOLD": [4],
        "HOLDING_BONUS_SCORE": [10, 20],
        "MA_TYPE": ["SMA", "EMA", "ALMA", "HMA"],
        "MA_MONTHS": [6, 9, 12, 18, 24],
        "RSI_LIMIT": [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100],
    },
    "kor_us": {
        "BENCHMARK": {"ticker": "379800", "name": "KODEX 미국S&P500"},
        "TOP_N_HOLD": [4],
        "HOLDING_BONUS_SCORE": [10, 20],
        "MA_TYPE": ["SMA", "EMA", "ALMA", "HMA"],
        "MA_MONTHS": [6, 9, 12, 18, 24],
        "RSI_LIMIT": [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100],
    },
    "aus": {
        "BENCHMARK": {"ticker": "IVV", "name": "iShares S&P 500"},
        "TOP_N_HOLD": [7],
        "HOLDING_BONUS_SCORE": [10, 20],
        "MA_TYPE": ["SMA", "EMA", "ALMA", "HMA"],
        "MA_MONTHS": [6, 9, 12, 18, 24],
        "RSI_LIMIT": [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100],
    },
    "us": {
        "BENCHMARK": {"ticker": "VOO", "name": "뱅가드 S&P500"},
        "TOP_N_HOLD": [5],
        "HOLDING_BONUS_SCORE": [10, 20],
        "MA_TYPE": ["SMA", "EMA", "ALMA", "HMA"],
        "MA_MONTHS": [6, 9, 12, 18, 24],
        "RSI_LIMIT": [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100],
    },
    "kor": {
        "BENCHMARK": {"ticker": "005930", "name": "삼성전자"},
        "TOP_N_HOLD": [3],
        "HOLDING_BONUS_SCORE": [10, 20],
        "MA_TYPE": ["SMA", "EMA", "ALMA", "HMA"],
        "MA_MONTHS": [6, 9, 12, 18, 24],
        "RSI_LIMIT": [80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100],
    },
}
