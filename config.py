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

# 네이버 벌크 종목 시세 정보 (KOSPI/KOSDAQ)
NAVER_STOCK_MARKET_VALUE_URL = "https://m.stock.naver.com/api/stocks/marketValue/{market}"
NAVER_STOCK_MARKET_VALUE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/04.1",
    "Referer": "https://m.stock.naver.com/",
    "Accept": "application/json, text/plain, */*",
}

# 토스증권 API 설정 (미국 주식 실시간 / 시장지표 mini-chart)
TOSS_INVEST_API_BASE_URL = "https://wts-info-api.tossinvest.com"
TOSS_INVEST_CERT_API_BASE_URL = "https://wts-cert-api.tossinvest.com"
TOSS_INVEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Referer": "https://tossinvest.com/",
    "Origin": "https://tossinvest.com",
    "Content-Type": "application/json",
}


# 네이버 미국 개별주 시가총액/업종 정보
NAVER_US_STOCK_MARKET_VALUE_URL = "https://stock.naver.com/api/foreign/market/stock/global"

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

# Hyperliquid (24시간 토큰화 주식 시세) — /hyperliquid 화면.
# 빌더 DEX `xyz` 가 SMSN(삼성전자)/SKHX(SK하이닉스)/MU(마이크론) 등 perp 을 24h 거래한다.
# 가격은 USD. 한국 종목은 환율로 KRW 환산해 실제(KRX) 가와 비교하고, 미국 종목은 USD 그대로 비교.
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_DEX = "xyz"
# live-24h 슬랙 알림: 최근 1시간 |변동률| 이 이 값(%) 이상인 종목이 있으면 @channel 핑.
LIVE_24H_ALERT_PCT = 3.0
# type: "stock"=개별주(가격 USD, 한국은 환율로 KRW 환산 / 실제가=네이버·토스),
#       "index"=지수(포인트 그대로, 통화 없음 / 실제가=네이버 KR 지수 또는 야후 심볼)
HYPERLIQUID_SYMBOLS = [
    {
        "symbol": "SKHX",
        "name": "SK하이닉스",
        "type": "stock",
        "country": "kor",
        "actual_ticker": "000660",
    },
    {
        "symbol": "SMSN",
        "name": "삼성전자",
        "type": "stock",
        "country": "kor",
        "actual_ticker": "005930",
    },
    {
        "symbol": "MU",
        "name": "마이크론",
        "type": "stock",
        "country": "us",
        "actual_ticker": "MU",
    },
]


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

# 지표 계산에 필요한 절대 최소 거래일 수 (항상 적용)
# ENABLE_DATA_SUFFICIENCY_CHECK = True  → 엄격 기준 적용
# ENABLE_DATA_SUFFICIENCY_CHECK = False → 이 값만 체크 (신규 상장 ETF 조기 포착용)
# 5일(1주) 미만 데이터는 추세 판단이 불가하므로 제외
MIN_TRADING_DAYS = 5

# -----------------------------------------------------------------------
# 시장지수 추세 (/market-trend)
# -----------------------------------------------------------------------
# 백엔드(추세점수 정규화 / 레짐 판정)에서 쓰는 단일 진실 소스.

# 추세점수 정규화 앵커 퍼센타일. 12개월 괴리율의 상위 P%를 +100, 하위 (100-P)%를 −100 으로
# 환산한다. 예) 95 → 상위 5%/하위 5%, 90 → 상위 10%/하위 10%. 값↓ = 100%/10% 에 더 쉽게 도달.
MARKET_TREND_SCORE_ANCHOR_PERCENTILE = 90

# MA20/60 레짐 판정 설정. 지수별 N일 확인만 사람이 조정하는 운영 튜닝값이다.
MARKET_TREND_REGIME_MA_SHORT = 20
MARKET_TREND_REGIME_MA_LONG = 60
MARKET_TREND_REGIME_CONFIRM_DAYS: dict[str, int] = {
    "^KS11": 2,
    "^KS200": 2,
    "^DJI": 1,
    "^GSPC": 4,
    "^NDX": 4,
    "^SOX": 2,
}

# 슈퍼트렌드(SuperTrend) 지표 설정.
# 차트 보조선/화살표 표시 전용이다. 레짐 판정과 탑픽 현금비중에는 쓰지 않는다.
# ATR 계산 기간(PERIOD)은 전 지수 공통. 곱수(MULTIPLIER)는 지수마다 개별 등록한다.
MARKET_TREND_SUPERTREND_PERIOD = 10

# 지수별 슈퍼트렌드 곱수 (yf_ticker → multiplier). 사용하는 모든 지수를 반드시 등록한다.
# 값↑=방향 전환이 뜸해져 휩쏘↓(지연 없음) / 값↓=추세 전환에 민감. 지수마다 변동성이 달라 개별 설정.
# yf_ticker: ^KS11=코스피, ^KS200=코스피200, ^DJI=다우존스, ^GSPC=S&P500, ^NDX=나스닥100, ^SOX=필라델피아 반도체.
MARKET_TREND_SUPERTREND_MULTIPLIER: dict[str, float] = {
    "^KS11": 1.5,    # 코스피 (빠른 대응이 필요)
    "^KS200": 1.5,   # 코스피200 (빠른 대응이 필요)
    "^DJI": 3.0,     # 다우존스
    "^GSPC": 3.0,    # S&P500
    "^NDX": 3.0,     # 나스닥100
    "^SOX": 1.5,     # 필라델피아 반도체 (빠른 대응이 필요)
}

# -----------------------------------------------------------------------
# 백테스트 파라미터 스윕 설정
# -----------------------------------------------------------------------
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
