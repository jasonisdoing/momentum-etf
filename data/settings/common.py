CACHE_START_DATE = "2022-01-01"
SELECTED_STRATEGY = "maps"

MARKET_REGIME_FILTER_TICKER_MAIN = "^GSPC"
MARKET_REGIME_FILTER_TICKERS_AUX = ["^IXIC", "NQ=F", "^DJI"]

MARKET_REGIME_FILTER_COUNTRY = "us"
MARKET_REGIME_FILTER_DELAY_DAY = 1
MARKET_REGIME_FILTER_MA_PERIOD = 20

# RSI 점수 정규화 설정 (0~100 스케일)
RSI_NORMALIZATION_CONFIG = {
    "enabled": True,
    "oversold_threshold": 30.0,  # 과매도 기준 (RSI < 30)
    "overbought_threshold": 70.0,  # 과매수 기준 (RSI > 70)
}

# RSI 계산 설정
RSI_CALCULATION_CONFIG = {
    "period": 14,  # RSI 계산 기간
    "ema_smoothing": 3.0,  # EMA 평활화 계수
}

# RSI 과매수 매도는 항상 활성화됨
# overbought_sell_threshold는 계좌별 strategy.tuning.OVERBOUGHT_SELL_THRESHOLD에서 설정 (정수)

# 백테스트 슬리피지 설정 (%)
# 매수/매도 시 시초가 대비 불리한 가격으로 체결된다고 가정
BACKTEST_SLIPPAGE = {
    "kor": {
        "buy_pct": 0.5,  # 매수 시 시초가 + 0.5%
        "sell_pct": 0.5,  # 매도 시 시초가 - 0.5%
    },
    "aus": {
        "buy_pct": 1.0,  # 매수 시 시초가 + 1.0%
        "sell_pct": 1.0,  # 매도 시 시초가 - 1.0%
    },
    "us": {
        "buy_pct": 0.3,  # 매수 시 시초가 + 0.3%
        "sell_pct": 0.3,  # 매도 시 시초가 - 0.3%
    },
}
