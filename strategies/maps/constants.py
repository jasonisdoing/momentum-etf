"""MAPS(Moving Average Position Score) 전략에서 사용하는 상수 모음."""

from __future__ import annotations

BACKTEST_STATUS_LIST = {
    "HOLD": {
        "display_name": "",
        "order": 10,
        "is_recommendation": False,
        "show_slack": True,
        # [User Request] 보유 종목 하이라이트 (연한 초록)
        "background": "#d0f0c0",
    },
    "BUY_REPLACE": {
        "display_name": "<🔄 교체매수>",
        "order": 25,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#81c147",
    },
    "SELL_REPLACE": {
        "display_name": "<🔄 교체매도>",
        "order": 26,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#ffc1cc",
    },
    "SELL": {
        "display_name": "<🔻 매도>",
        "order": 27,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#ffc1cc",
    },
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 31,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#81c147",
    },
    "WAIT": {
        "display_name": "",
        "order": 50,
        "is_recommendation": False,
        "show_slack": False,
        "background": "#f0f0f0",
    },
}


def _normalize_display_label(raw: str | None) -> str:
    value = str(raw or "").strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value


_DECISION_MESSAGE_OVERRIDES: dict[str, str] = {
    "BUY": "✅ 신규 매수",
    "SELL": "🔻 매도",
    "BUY_REPLACE": "🔄 교체매수",
    "SELL_REPLACE": "🔄 교체매도",
}

PENDING_ACTION_MESSAGES: dict[str, str] = {
    "BUY": "[예정] 신규 매수 신호",
    "BUY_REPLACE": "[예정] 교체매수 신호",
    "SELL": "[예정] 매도 신호",
    "SELL_REPLACE": "[예정] 교체매도 신호",
    "SELL_REBALANCE": "",
}


DECISION_MESSAGES = {
    key: _normalize_display_label(cfg.get("display_name"))
    for key, cfg in BACKTEST_STATUS_LIST.items()
    if isinstance(cfg, dict) and cfg.get("display_name")
}

for override_key, override_value in _DECISION_MESSAGE_OVERRIDES.items():
    DECISION_MESSAGES[override_key] = override_value

DECISION_MESSAGES = {key: value for key, value in DECISION_MESSAGES.items() if value}

DECISION_MESSAGES["NEW_BUY"] = DECISION_MESSAGES.get("BUY", "✅ 신규 매수")

DECISION_NOTES = {
    "INSUFFICIENT_CASH": "현금 부족",
    "NO_PRICE": "가격 정보 없음",
    "REPLACE_SELL": "교체 매도",
    "PRICE_DATA_FAIL": "가격 데이터 조회 실패",
    "DATA_INSUFFICIENT": "⚠️ 거래일 부족",
}

__all__ = [
    "BACKTEST_STATUS_LIST",
    "DECISION_MESSAGES",
    "DECISION_NOTES",
    "PENDING_ACTION_MESSAGES",
]
