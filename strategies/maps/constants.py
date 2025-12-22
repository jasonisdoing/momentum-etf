"""MAPS(Moving Average Position Score) 전략에서 사용하는 상수 모음."""

from __future__ import annotations

DECISION_CONFIG = {
    "HOLD_CORE": {
        "display_name": "<🔒 핵심보유>",
        "order": 1,
        "is_recommendation": False,
        "show_slack": True,
        "background": "#e8f5e9",
    },
    "HOLD": {
        "display_name": "<💼 보유>",
        "order": 10,
        "is_recommendation": False,
        "show_slack": True,
        "background": None,
    },
    "CUT_STOPLOSS": {
        "display_name": "<🚨 손절매도>",
        "order": 11,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_TREND": {
        "display_name": "<📉 추세이탈 매도>",
        "order": 21,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_RSI": {
        "display_name": "<🔥 RSI 과매수 매도>",
        "order": 22,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#ffb6c1",
    },
    "SELL_REPLACE": {
        "display_name": "<🔄 교체매도>",
        "order": 23,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#ffc1cc",
    },
    "BUY_REPLACE": {
        "display_name": "<🔄 교체매수>",
        "order": 30,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#81c147",
    },
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 31,
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
        "order": 100,
        "is_recommendation": False,
        "show_slack": True,
        "background": "#a0a0a0",
    },
}


def _normalize_display_label(raw: str | None) -> str:
    value = str(raw or "").strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value


_DECISION_MESSAGE_OVERRIDES: dict[str, str] = {
    "BUY": "✅ 신규 매수",
    "SOLD": "🔚 매도 완료",
    "BUY_REPLACE": "🔄 교체매수",
    "SELL_REPLACE": "🔄 교체매도",
    "SELL_RSI": "🔥 RSI 과매수 매도",
    "HOLD_CORE": "🔒 핵심 보유",
}


DECISION_MESSAGES = {
    key: _normalize_display_label(cfg.get("display_name"))
    for key, cfg in DECISION_CONFIG.items()
    if isinstance(cfg, dict) and cfg.get("display_name")
}

for override_key, override_value in _DECISION_MESSAGE_OVERRIDES.items():
    DECISION_MESSAGES[override_key] = override_value

DECISION_MESSAGES = {key: value for key, value in DECISION_MESSAGES.items() if value}

DECISION_MESSAGES["NEW_BUY"] = DECISION_MESSAGES.get("BUY", "✅ 신규 매수")

DECISION_NOTES = {
    "INSUFFICIENT_CASH": "현금 부족",
    "NO_PRICE": "가격 정보 없음",
    "TREND_BREAK": "추세 이탈",
    "REPLACE_SELL": "교체 매도",
    "PRICE_DATA_FAIL": "가격 데이터 조회 실패",
    "REPLACE_SCORE": "교체 {replace_score:.1f}점수 미만",
    "DATA_INSUFFICIENT": "⚠️ 거래일 부족",
    # 템플릿
    "COOLDOWN_GENERIC": "쿨다운 {days}일 대기중",
}

__all__ = [
    "DECISION_CONFIG",
    "DECISION_MESSAGES",
    "DECISION_NOTES",
]
