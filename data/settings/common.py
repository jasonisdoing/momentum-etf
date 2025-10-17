CACHE_START_DATE = "2020-01-01"
SELECTED_STRATEGY = "maps"

MARKET_REGIME_FILTER_TICKER_MAIN = "^GSPC"
MARKET_REGIME_FILTER_TICKERS_AUX = ["^IXIC", "NQ=F", "^DJI"]

MARKET_REGIME_FILTER_COUNTRY = "us"
MARKET_REGIME_FILTER_DELAY_DAY = 1
MARKET_REGIME_FILTER_MA_PERIOD = 20

# MAPS 점수 정규화 설정 (0~100 스케일)
MAPS_SCORE_NORMALIZATION_CONFIG = {
    "enabled": True,
    "eligibility_threshold": 0.0,  # MA와 동일 = 0점, MA 아래 = 0점
    "max_bound": 30.0,  # MA 대비 +30% 이상 = 100점
}

# RSI 점수 정규화 설정 (0~100 스케일)
RSI_NORMALIZATION_CONFIG = {
    "enabled": True,
    "oversold_threshold": 30.0,  # 과매도 기준 (RSI < 30)
    "overbought_threshold": 70.0,  # 과매수 기준 (RSI > 70)
}

# RSI 계산 설정
RSI_CALCULATION_CONFIG = {
    "period": 14,  # RSI 계산 기간
    "ema_smoothing": 2.0,  # EMA 평활화 계수
}

# 종합 점수 계산 설정
COMPOSITE_SCORE_CONFIG = {
    "enabled": True,
    "method": "rsi_adjusted",  # "weighted_average" 또는 "rsi_adjusted"
    # weighted_average 방식 설정
    "maps_weight": 0.7,  # MAPS 점수 가중치 (0.0 ~ 1.0)
    "rsi_weight": 0.3,  # RSI 점수 가중치 (0.0 ~ 1.0)
    # rsi_adjusted 방식 설정
    "rsi_multiplier_min": 0.3,  # RSI=0 (과매수)일 때 MAPS 점수 배율
    "rsi_multiplier_max": 1.2,  # RSI=100 (과매도)일 때 MAPS 점수 배율
}
