"""Global configuration values used across the project."""

CACHE_START_DATE = "2022-01-01"
SELECTED_STRATEGY = "maps"

MARKET_REGIME_FILTER_TICKER_MAIN = "^GSPC"
MARKET_REGIME_FILTER_TICKERS_AUX = ["^IXIC", "NQ=F", "^DJI"]

MARKET_REGIME_FILTER_COUNTRY = "us"
MARKET_REGIME_FILTER_DELAY_DAY = 1
MARKET_REGIME_FILTER_MA_PERIOD = 50

# RSI normalization settings (0~100 scale)
RSI_NORMALIZATION_CONFIG = {
    "enabled": True,
    "oversold_threshold": 30.0,  # currently unused
    "overbought_threshold": 70.0,  # used as sell trigger
}

# RSI calculation parameters
RSI_CALCULATION_CONFIG = {
    "period": 15,
    "ema_smoothing": 2.0,
}

# Backtest slippage assumptions (%)
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

# Real-time ETF quote source selection, Allowed values: "Price", "Nav"
KOR_REALTIME_ETF_PRICE_SOURCE = "Price"
# KOR_REALTIME_ETF_PRICE_SOURCE = "Nav"
