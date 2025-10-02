"""MAPS(Moving Average Position Score) 전략에서 사용하는 상수 모음."""

from __future__ import annotations

DECISION_CONFIG = {
    "HOLD": {
        "display_name": "<💼 보유>",
        "order": 1,
        "is_recommendation": False,
        "show_slack": True,
        "background": None,
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
        "background": "#ffc1cc",
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
        "background": "#81c147",
    },
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 21,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#81c147",
    },
    "WAIT": {
        "display_name": "<⏳ 대기>",
        "order": 50,
        "is_recommendation": False,
        "show_slack": False,
        "background": "#f0f0f0",
    },
    "SOLD": {
        "display_name": "<✅ 매도 완료>",
        "order": 50,
        "is_recommendation": False,
        "show_slack": True,
        "background": "#f0f0f0",
    },
}

DECISION_MESSAGES = {
    # 매수
    "NEW_BUY": "✅ 신규 매수",
    # 매도
    "SOLD": "🔚 매도 완료",
}

DECISION_NOTES = {
    "CATEGORY_DUP": "카테고리 중복",
    "PORTFOLIO_FULL": "포트폴리오 가득 참",
    "INSUFFICIENT_CASH": "현금 부족",
    "NO_PRICE": "가격 정보 없음",
    "RISK_OFF": "시장 위험 회피",
    "RISK_OFF_SELL": "시장위험회피 매도",
    "TREND_BREAK": "추세 이탈",
    "REPLACE_SELL": "교체 매도",
    "LOCKED_HOLD": "신호와 상관없이 보유",
    "PRICE_DATA_FAIL": "가격 데이터 조회 실패",
    # 템플릿
    "COOLDOWN_GENERIC": "쿨다운 {days}일 대기중",
    "COOLDOWN_WITH_ACTION": "쿨다운 {days}일 대기중 ({action} {date})",
}

__all__ = [
    "DECISION_CONFIG",
    "DECISION_MESSAGES",
    "DECISION_NOTES",
]
