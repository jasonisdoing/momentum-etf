"""Momentum 전략에서 사용하는 상수 모음."""

from __future__ import annotations

DECISION_CONFIG = {
    "HOLD": {
        "display_name": "<💼 보유>",
        "order": 1,
        "is_recommendation": False,
        "show_slack": True,
    },
    "CUT_STOPLOSS": {
        "display_name": "<🚨 손절매도>",
        "order": 10,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_TREND": {
        "display_name": "<📉 추세이탈 매도>",
        "order": 11,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REPLACE": {
        "display_name": "<🔄 교체매도>",
        "order": 12,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REGIME_FILTER": {
        "display_name": "<🛡️ 시장위험회피 매도>",
        "order": 15,
        "is_recommendation": True,
        "show_slack": True,
    },
    "BUY_REPLACE": {
        "display_name": "<🔄 교체매수>",
        "order": 20,
        "is_recommendation": True,
        "show_slack": True,
    },
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 21,
        "is_recommendation": True,
        "show_slack": True,
    },
    "WAIT": {
        "display_name": "<⏳ 대기>",
        "order": 50,
        "is_recommendation": False,
        "show_slack": False,
    },
    "SOLD": {
        "display_name": "<✅ 매도 완료>",
        "order": 60,
        "is_recommendation": False,
        "show_slack": True,
    },
}

# 거래소 최소 잔량 처리에 사용되는 코인 잔고 임계값
COIN_ZERO_THRESHOLD = 1e-9

__all__ = ["DECISION_CONFIG", "COIN_ZERO_THRESHOLD"]
