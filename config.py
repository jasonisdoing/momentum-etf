"""프로젝트 전역에서 사용하는 설정 값 정의."""

import json
from pathlib import Path

CACHE_START_DATE = "2024-01-01"
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

# 토스증권 API 설정 (미국 주식 실시간)
TOSS_INVEST_API_BASE_URL = "https://wts-info-api.tossinvest.com"
TOSS_INVEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://tossinvest.com/",
    "Origin": "https://tossinvest.com",
    "Content-Type": "application/json",
}


# 네이버 미국 개별주 시가총액/업종 정보
NAVER_US_STOCK_MARKET_VALUE_URL = "https://stock.naver.com/api/foreign/market/stock/global"

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

# 지원 이동평균(MA) 타입 — 시스템 전체 단일 진실 소스.
# 백엔드(rankings/market_trend/pool_settings 검증·옵션) + 프론트(MA 드롭다운)에서 모두 이 값만 본다.
# 프론트에는 API 응답(rank: ma_type_options / market-trend defaults: ma_types)으로 전달된다.
ALLOWED_MA_TYPES = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA", "ALMA"]

# 1개월 = 20 거래일 (MA 개월 → 거래일 변환에 사용)
TRADING_DAYS_PER_MONTH = 20

# 지표 계산에 필요한 절대 최소 거래일 수 (MA 타입 무관, 항상 적용)
# ENABLE_DATA_SUFFICIENCY_CHECK = True  → MA 타입별 엄격 기준 적용 (60~120일)
# ENABLE_DATA_SUFFICIENCY_CHECK = False → 이 값만 체크 (신규 상장 ETF 조기 포착용)
# 5일(1주) 미만 데이터는 추세 판단이 불가하므로 제외
MIN_TRADING_DAYS = 5

# -----------------------------------------------------------------------
# 시장지수 추세 / 권장 투자 비율 (/market-trend)
# -----------------------------------------------------------------------
# 백엔드(추세점수 정규화) + 프론트(MA 선택/권장 투자 매핑)에서 함께 쓰는 단일 진실 소스.
# 프론트에는 /internal/market-trend/defaults 응답으로 전달된다.

# 추세점수 정규화 앵커 퍼센타일. 12개월 괴리율의 상위 P%를 +100, 하위 (100-P)%를 −100 으로
# 환산한다. 예) 95 → 상위 5%/하위 5%, 90 → 상위 10%/하위 10%. 값↓ = 100%/10% 에 더 쉽게 도달.
MARKET_TREND_SCORE_ANCHOR_PERCENTILE = 90

# MA 기간 선택 드롭다운 상한 (개월). 1 ~ 이 값.
MARKET_TREND_MA_MONTHS_MAX = 12

# 권장 투자비율 매핑 (추세점수 → 투자%). 기존 운용 직관을 연속화한 앵커:
#   중립(점수 0) = NEUTRAL_INVEST(%), 점수 +100 = NEUTRAL+UP_SPAN, 점수 −100 = NEUTRAL−DOWN_SPAN
#   점수 ≥ 0 : 투자% = NEUTRAL + (점수/100) × UP_SPAN
#   점수 < 0 : 투자% = NEUTRAL + (점수/100) × DOWN_SPAN
# 기본값 70/30/60 → 중립 70%, 완전상승 100%, 완전하락 10%.
MARKET_TREND_ALLOC_NEUTRAL_INVEST = 70
MARKET_TREND_ALLOC_UP_SPAN = 30
MARKET_TREND_ALLOC_DOWN_SPAN = 60

# 레짐별 투자 상한(%). 점수기반 권장투자에 min() 으로 천장을 씌운다.
#   최종 투자% = min(점수기반%, 해당 레짐 상한)
# 점수는 레벨(MA 대비 위치)이라 고점에서 +100(=100%)이 되는데, 그때 레짐이 약화(조정)면
# 천장을 눌러 "꼭지 풀투자"를 막는다.
# 실효 캡 2개만 둔다 — 상승(MA 위+강화)·진정(MA 아래+회복)은 base 가 이미 그 아래라
# 천장이 안 걸리므로(no-op) 파라미터에서 제외했다.
#   중립조정(decel_up): MA 위 + 약화(천장권) — base 최대 100 을 눌러줌
#   하락(accel_down) : MA 아래 + 약화      — MA 근처 base 를 방어적으로 낮춤
MARKET_TREND_ALLOC_CAP_DECEL_UP = 90
MARKET_TREND_ALLOC_CAP_ACCEL_DOWN = 70

# 레짐(가속/감속) 판정: 최근 4주 평균 비교 대신 추세%의 회귀 기울기 + 데드밴드(히스테리시스).
#   최근 SLOPE_WINDOW 거래일 추세%에 최소제곱 직선을 적합해 기울기(%/일)를 구하고,
#   기울기 > +DEADBAND → 강화, < −DEADBAND → 약화, 그 사이면 직전 상태 유지(라벨 휩소 차단).
# 값↑(WINDOW) = 더 매끈/둔감, 값↑(DEADBAND) = 라벨이 덜 바뀜.
MARKET_TREND_REGIME_SLOPE_WINDOW = 20
MARKET_TREND_REGIME_SLOPE_DEADBAND = 0.05

# -----------------------------------------------------------------------
# 백테스트 파라미터 스윕 설정
# -----------------------------------------------------------------------
BACKTEST_START_DATE = "2025-06-01"
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
    "all": {
        "BENCHMARK": {"ticker": "456600", "name": "TIME 글로벌AI인공지능액티브"},
        "TOP_N_HOLD": [4],
        "HOLDING_BONUS_SCORE": [10],
        "MA_TYPE": ["ALMA"],
        "MA_MONTHS": [6],
        "RSI_LIMIT": [100],
    },
    "kor_kr": {
        "BENCHMARK": {"ticker": "069500", "name": "KODEX 200"},
        "TOP_N_HOLD": [5],
        "HOLDING_BONUS_SCORE": [0, 10, 20],
        "MA_TYPE": ["ALMA"],
        "MA_MONTHS": [3, 6, 9, 12],
        "RSI_LIMIT": [100],
    },
    "kor_us": {
        "BENCHMARK": {"ticker": "379800", "name": "KODEX 미국S&P500"},
        "TOP_N_HOLD": [3],
        "HOLDING_BONUS_SCORE": [0, 10, 20],
        "MA_TYPE": ["ALMA"],
        "MA_MONTHS": [3, 6, 9, 12],
        "RSI_LIMIT": [100],
    },
    "aus": {
        "BENCHMARK": {"ticker": "IVV", "name": "iShares S&P 500"},
        "TOP_N_HOLD": [8],
        "HOLDING_BONUS_SCORE": [0, 10, 20],
        "MA_TYPE": ["ALMA"],
        "MA_MONTHS": [3, 6, 9, 12],
        "RSI_LIMIT": [100],
    },
    "us": {
        "BENCHMARK": {"ticker": "QQQ", "name": "인베스코 QQQ ETF"},
        "TOP_N_HOLD": [5],
        "HOLDING_BONUS_SCORE": [0, 10, 20],
        "MA_TYPE": ["ALMA"],
        "MA_MONTHS": [3, 6, 9, 12],
        "RSI_LIMIT": [100],
    },
    "kor": {
        "BENCHMARK": {"ticker": "005930", "name": "삼성전자"},
        "TOP_N_HOLD": [4],
        "HOLDING_BONUS_SCORE": [0, 10, 20],
        "MA_TYPE": ["ALMA"],
        "MA_MONTHS": [3, 6, 9, 12],
        "RSI_LIMIT": [100],
    },
}
