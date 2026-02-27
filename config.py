"""프로젝트 전역에서 사용하는 설정 값 정의."""

CACHE_START_DATE = "2024-01-01"
SLACK_CHANNEL = "C0A0X2LTS3X"


# -----------------------------------------------------------------------
# 버킷(Bucket) 설정 및 스타일링
# -----------------------------------------------------------------------
BUCKET_CONFIG = {
    1: {"name": "1. 모멘텀", "bg_color": "#ffebee", "text_color": "#c62828"},
    2: {"name": "2. 혁신기술", "bg_color": "#fff3e0", "text_color": "#ef6c00"},
    3: {"name": "3. 시장지수", "bg_color": "#f3e5f5", "text_color": "#7b1fa2"},
    4: {"name": "4. 배당방어", "bg_color": "#e3f2fd", "text_color": "#1565c0"},
    5: {"name": "5. 대체헷지", "bg_color": "#e8f5e9", "text_color": "#2e7d32"},
}

BUCKET_MAPPING = {k: v["name"] for k, v in BUCKET_CONFIG.items()}
BUCKET_REVERSE_MAPPING = {v: k for k, v in BUCKET_MAPPING.items()}
BUCKET_OPTIONS = list(BUCKET_MAPPING.values())

# 네이버 금융 API 설정
NAVER_FINANCE_ETF_API_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
NAVER_FINANCE_STOCK_POLLING_URL = "https://polling.finance.naver.com/api/realtime"
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

# 1개월 = 20 거래일 (MA 개월 → 거래일 변환에 사용)
TRADING_DAYS_PER_MONTH = 20

# 지표 계산에 필요한 절대 최소 거래일 수 (MA 타입 무관, 항상 적용)
# ENABLE_DATA_SUFFICIENCY_CHECK = True  → MA 타입별 엄격 기준 적용 (60~120일)
# ENABLE_DATA_SUFFICIENCY_CHECK = False → 이 값만 체크 (신규 상장 ETF 조기 포착용)
# 5일(1주) 미만 데이터는 추세 판단이 불가하므로 제외
MIN_TRADING_DAYS = 5

# 리밸런싱 주기 설정 (전역 공통)
# DAILY, WEEKLY, TWICE_A_MONTH, MONTHLY, QUARTERLY 중 선택
REBALANCE_MODE = "TWICE_A_MONTH"

# 튜닝 최적화 지표 (전역 공통)
# CAGR, SHARPE, SDR 중 선택
OPTIMIZATION_METRIC = "CAGR"

# 리밸런싱 교체 후보군(Candidate) 선정 임계값(%)
# 대기 1등 종목 점수를 기준으로, 지정한 퍼센트(%) 이내의 점수 하락폭을 가진 종목들을 매수 후보로 노출합니다.
# 점수 스케일(MA TYPE 등)수준에 구애받지 않도록 상대적 비율(%)로 계산됩니다.
# 예: 25.0인 경우, 대기 1등 점수에서 26% 하락한 점수까지 커트라인으로 인정하여 묶어서 추천합니다.
REBALANCE_CANDIDATE_THRESHOLD = 25.0
