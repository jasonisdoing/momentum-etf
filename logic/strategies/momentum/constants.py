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
        "order": 50,
        "is_recommendation": False,
        "show_slack": True,
    },
}

# 거래소 최소 잔량 처리에 사용되는 코인 잔고 임계값
COIN_ZERO_THRESHOLD = 1e-9

DECISION_MESSAGES = {
    # 매수
    "NEW_BUY": "✅ 신규 매수",
    # {amount}: 금액(예: 123만원)
    "PARTIAL_BUY": "🌗 부분 매수({amount})",
    # 매도
    # {amount}: 금액(예: 123만원)
    "PARTIAL_SELL": "⚠️ 부분 매도 ({amount})",
    "FULL_SELL": "🔚 매도 완료",
}

DECISION_NOTES = {
    "CATEGORY_DUP": "카테고리 중복",
    "PORTFOLIO_FULL": "포트폴리오 가득 참",
    "INSUFFICIENT_CASH": "현금 부족",
    "NO_PRICE": "가격 정보 없음",
    "RISK_OFF": "시장 위험 회피",
    "RISK_OFF_SELL": "시장위험회피 매도",
    "LOCKED_HOLD": "신호와 상관없이 보유",
    "PRICE_DATA_FAIL": "가격 데이터 조회 실패",
    # 템플릿
    "COOLDOWN_GENERIC": "쿨다운 {days}일 대기중",
    "COOLDOWN_WITH_ACTION": "쿨다운 {days}일 대기중 ({action} {date})",
}

__all__ = [
    "DECISION_CONFIG",
    "COIN_ZERO_THRESHOLD",
    "DECISION_MESSAGES",
    "DECISION_NOTES",
]
