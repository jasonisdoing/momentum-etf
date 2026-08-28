"""프로젝트 전역에서 사용하는 설정 값 정의."""

import json
from pathlib import Path

SLACK_CHANNEL = "C0A0X2LTS3X"

# python scripts/update_market_calendars.py CACHE_START_DATE 변경시 실행
CACHE_START_DATE = "2018-12-31"

# 지표 기준 창(개월) — 고점 대비(%)·MDD·소르티노 등 화면 지표는 최근 이 기간을 기준으로 계산한다.
METRIC_WINDOW_MONTHS = 12

# 메모리 캐시 TTL(초) — 프로젝트의 모든 TTL 캐시는 이 4개 중 하나를 쓴다.
# 새 캐시를 만들 때 값을 직접 적지 말고 성격에 맞는 상수를 골라 쓴다.
CACHE_TTL_LIVE = 60  # 실시간성 필요 — 시세·설정·목록
CACHE_TTL_COMPUTE = 300  # 무거운 계산·집계 결과 — 랭킹·전략 현재 상태·외부 집계
CACHE_TTL_SLOW = 3600  # 느리게 변하는 외부 값 — 환율·심볼 해석
CACHE_TTL_META = 86400  # 사실상 고정된 메타 — ETF 기본 정보

# 백테스트 시작 자본 — **통화별**. 주식은 정수 주수로만 살 수 있어 백테스트도 정수로 굴리는데,
# "1주"가 의미를 가지려면 시작 자본이 있어야 한다(상대곡선만으로는 주수를 셀 수 없다).
#
# 계좌 총자산을 쓰지 않는다 — 백테스트·튜닝은 종목풀 단위로 돌고, 계좌가 연결되지 않은 풀도
# 있어 참조할 잔고가 없다. 계좌 잔고를 쓰면 입출금 때마다 같은 설정의 결과가 달라지기도 한다.
#
# 금액은 **실제 운용 규모에 맞춘다**. 정수 제약의 마찰(1주 값에 걸려 남는 현금)은 자본이
# 클수록 옅어져서, 넉넉히 잡으면 백테스트가 실제보다 낙관적으로 나온다.
BACKTEST_INITIAL_CAPITAL = {
    "KRW": 20_000_000,
    "USD": 15_000,
    "AUD": 20_000,
}

# 전략 이동평균 종류 — 추세선·이격도·순위 계산에 쓰는 이동평균. "SMA"(단순) 또는 "EMA"(지수).
# 이 값 하나로 시스템 전체의 이동평균 계산·표시 문구가 바뀐다
# MOVING_AVERAGE_TYPE = "SMA"
MOVING_AVERAGE_TYPE = "EMA"


# -----------------------------------------------------------------------
# 버킷(Bucket) 설정 및 스타일링
# -----------------------------------------------------------------------
# 버킷 테마의 단일 소스 — 웹(web/lib/bucket-theme.ts)이 정적 임포트하는 파일을
# 파이썬도 같이 읽는다 (예전 shared/ 사본은 중복이라 제거).
_BUCKET_THEME_PATH = Path(__file__).resolve().parent / "web" / "lib" / "bucket_theme.json"
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

# `premarket_open`·`aftermarket_close` 는 정규장 밖 거래 시간이다(현지 시각).
# 없는 시장(호주)은 키를 두지 않는다 — 값을 지어내면 화면이 없는 세션을 보여준다.
MARKET_SCHEDULES = {
    "kor": {
        "open": time(9, 0),
        "close": time(15, 30),
        "open_offset_minutes": 30,
        "close_offset_minutes": 30,
        "premarket_open": time(8, 0),
        "aftermarket_close": time(20, 0),
        "timezone": "Asia/Seoul",
        "name": "한국",
    },
    "us": {
        "open": time(9, 30),
        "close": time(16, 0),
        "open_offset_minutes": 30,
        "close_offset_minutes": 30,
        "premarket_open": time(4, 0),
        "aftermarket_close": time(20, 0),
        # 데이장(오버나이트) — 애프터가 끝난 20:00 부터 다음 거래일 새벽까지. 한국 주간에
        # 미국 주식이 거래되는 구간이다. 자정을 넘기므로 시작·종료가 다른 날짜에 있다.
        "daymarket_open": time(20, 0),
        "daymarket_close": time(3, 50),
        "timezone": "America/New_York",
        "name": "미국",
    },
    "au": {
        "open": time(10, 0),
        "close": time(16, 0),
        "open_offset_minutes": 30,
        "close_offset_minutes": 30,
        "timezone": "Australia/Sydney",
        "name": "호주",
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
# 전략 공용 셀렉트 선택지
# -----------------------------------------------------------------------
# 모멘텀·신고가가 **같은 의미**로 쓰는 값의 단일 소스. 전략마다 상수를 따로 두면
# 한쪽만 고쳐져 화면 셀렉트·튜닝 축·저장 검증이 전략별로 갈린다 — 실제로 종목 수에
# 2 를 넣을 때 모멘텀만 바뀌어 두 화면이 달라졌다.
# 여기에 두는 기준은 **의미가 같은가** 다. 값이 우연히 같아도 뜻이 다르면
# (모멘텀의 주중 손절 vs 신고가의 손절선) 각 전략에 남긴다.
# 화면 셀렉트는 API 응답으로 받은 목록만 렌더한다(프론트에 복사본을 두지 않는다).

# 종목 수(동시 보유 슬롯) — 저장 검증도 이 목록으로 한다(별도 범위 없음).
TOP_N_OPTIONS: tuple[int, ...] = (5, 6, 8, 10)

# 한 업종에서 담을 수 있는 최대 종목 수. None = 제한 없음.
MAX_PER_INDUSTRY_OPTIONS: tuple[int | None, ...] = (1, 2, 3, None)

# 손절 기준(%) — 진입가·평단 대비 하락률. 세 군데가 같은 목록을 쓴다:
#   · 신고가 전략의 손절선          (new_high_service.STOP_LOSS_OPTIONS)
#   · 모멘텀 전략의 주중 손절        (momentum_service.INTRAWEEK_STOP_OPTIONS — 여기에 '없음' 추가)
#   · 종목풀의 보유종목 손절 알림 기준 (pool_settings_store.STOPLOSS_PCT_OPTIONS)
# '없음'을 허용할지는 쓰는 쪽이 정한다 — 목록 자체는 하나다.
STOP_LOSS_PCT_OPTIONS: tuple[float, ...] = (-7.0, -10.0)

# 거래대금 급증 하한 — 평소 대비 몇 배 이상이어야 신호로 인정할지. None = 제한 없음.
MIN_VALUE_MULT_OPTIONS: tuple[float | None, ...] = (5.0, 2.0, 1.0, None)

# 편도 슬리피지(%) — 0.05 ~ 0.50, 0.05 단위. 종목풀 설정에서 고른다.
SLIPPAGE_PCT_OPTIONS: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)

# 종목풀 성격 — 'stock'(개별주) 또는 'etf'. 업종 개념이 있는 풀인지 화면들이 이 값으로 본다.
POOL_KIND_OPTIONS: tuple[str, ...] = ("stock", "etf")

# 기간 셀렉트(개월) — 시스템 전체가 이 목록 하나만 쓴다(전략·화면별로 따로 두지 않는다).
# 가격 캐시가 못 채우는 구간을 빼는 건 `pool_signal_backtest_service.get_month_options()`.
BACKTEST_MONTH_OPTIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 12, 24, 36, 48, 60)

# 종목풀 신호 백테스트의 보유일(거래일) 선택지.
FORWARD_DAY_OPTIONS: tuple[int, ...] = (5, 10, 20, 40, 60)

# 포트폴리오 전략 — 리밸런싱 주기와 기준.
# 주기: 이 간격의 첫 거래일에 한 번만 판정한다. 'none' 은 최초 매수 뒤 되돌리지 않는다.
REBALANCE_OPTIONS: tuple[str, ...] = ("monthly", "quarterly", "yearly", "none")
REBALANCE_LABELS: dict[str, str] = {
    "monthly": "매월",
    "quarterly": "분기",
    "yearly": "매년",
    "none": "안 함",
}

# 리밸런싱 기준(%p) — 목표 비중과 현재 비중의 **절대 차이**가 이만큼 벌어져야 되돌린다.
# 0 은 넣지 않는다: 차이가 있으면 항상 매매가 나와 가격이 조금만 움직여도 비용만 든다.
REBALANCE_BAND_PCT_OPTIONS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

# 이평선 일수 선택지 — **국가별**이다. 종목풀 설정·순위·종목풀 백테스트·모멘텀·
# 신고가(이탈선)·보유종목 알림이 전부 여기서 받는다(`utils/ma_options` 가 국가로 골라 준다).
# 한국은 거래일 기준 3·4.5·6·9개월(60·90·120·180), 미국·호주는 관례(50·100일선)에 맞춘
# 50·75·100·150 — 한국:미국 = 1.2 배로 칸마다 짝이 맞아 두 시장 결과를 같은 자리끼리
# 비교할 수 있다. 값을 늘릴 때 이 대응이 깨지지 않는지 함께 본다. 단기는 세 국가 공통.
SHORT_MA_DAYS_BY_COUNTRY: dict[str, tuple[int, ...]] = {
    "kor": (20, 30, 40, 120, 240),
    "us": (20, 30, 40),
    "au": (20, 30, 40),
}
LONG_MA_DAYS_BY_COUNTRY: dict[str, tuple[int, ...]] = {
    "kor": (60, 90, 120, 180, 240),
    "us": (50, 75, 100, 150),
    "au": (50, 75, 100, 150),
}

# -----------------------------------------------------------------------
# 시장지수 추세 (/market-trend)
# -----------------------------------------------------------------------
# 백엔드(추세점수 정규화 / 레짐 판정)에서 쓰는 단일 진실 소스.

# 추세점수 정규화 앵커 퍼센타일. 12개월 괴리율의 상위 P%를 +100, 하위 (100-P)%를 −100 으로
# 환산한다. 예) 95 → 상위 5%/하위 5%, 90 → 상위 10%/하위 10%. 값↓ = 100%/10% 에 더 쉽게 도달.
MARKET_TREND_SCORE_ANCHOR_PERCENTILE = 90

# 추세 점수용 이동평균 일수
MARKET_TREND_SCORE_MA_DAYS = 20

# 슈퍼트렌드(SuperTrend) 지표 설정.
# 차트 보조선/화살표 표시 전용이다. 레짐 판정과 현금비중에는 쓰지 않는다.
# ATR 계산 기간(PERIOD)은 전 지수 공통. 곱수(MULTIPLIER)는 지수마다 개별 등록한다.
MARKET_TREND_SUPERTREND_PERIOD = 10

# 지수별 슈퍼트렌드 곱수 (yf_ticker → multiplier). 사용하는 모든 지수를 반드시 등록한다.
# 값↑=방향 전환이 뜸해져 휩쏘↓(지연 없음) / 값↓=추세 전환에 민감. 지수마다 변동성이 달라 개별 설정.
# yf_ticker: ^KS11=코스피, ^KQ11=코스닥, ^DJI=다우존스, ^GSPC=S&P500,
#            ^NDX=나스닥100, ^SOX=필라델피아 반도체, NQ=F=나스닥100 선물.
MARKET_TREND_SUPERTREND_MULTIPLIER: dict[str, float] = {
    "^KS11": 1.5,  # 코스피 (빠른 대응이 필요)
    "^KQ11": 1.5,  # 코스닥 (빠른 대응이 필요)
    "^DJI": 3.0,  # 다우존스
    "^GSPC": 3.0,  # S&P500
    "^NDX": 3.0,  # 나스닥100
    "^SOX": 1.5,  # 필라델피아 반도체 (빠른 대응이 필요)
    "NQ=F": 3.0,  # 나스닥100 선물
    # 원/달러 — 변동성이 지수보다 낮아 같은 곱수라도 국면이 훨씬 길게 유지된다.
    # 24개월 실측 평균 지속일: 1.5배 19일 / 2.0배 35일 / 3.0배 58일.
    # 1.5 를 쓰면 코스피(12일)와 S&P500(28일) 사이에 들어와 다른 지수와 비슷하게 읽힌다.
    "KRW=X": 1.5,  # 미국달러
}

# ADR(등락비율, Advance-Decline Ratio) 설정.
# ADR = 기간 내 상승종목수 합 / 하락종목수 합 × 100. 지수는 대형주 몇 종목에 끌려가지만
# ADR 은 '얼마나 많은 종목이 함께 오르는가'(시장 폭)를 본다. 지수가 오르는데 ADR 이 빠지면
# 상승 종목이 좁아지는 국면이라 모멘텀 전략에 불리하다.
MARKET_ADR_WINDOW_DAYS = 20

# 과매수/과매도 판정 경계. 한국 시장에서 통용되는 관례값이다.
MARKET_ADR_OVERHEATED = 120.0
MARKET_ADR_OVERSOLD = 75.0

# 강세/약세를 가르는 중립선. ADR 100 = 상승 종목수와 하락 종목수가 같다.
MARKET_ADR_NEUTRAL = 100.0

# ADR 대상 종목 수 — 시가총액 상위 N 종목. 전 종목이 아니라 상위 N 인 이유는
# 동전주·거래정지 종목의 잡음을 빼고 매일 같은 기준으로 비교하기 위해서다.
MARKET_ADR_UNIVERSE_SIZE = 200
