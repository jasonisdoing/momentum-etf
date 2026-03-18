"""RANK(Moving Average Position Score) 전략에서 사용하는 상수 모음."""

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
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 20,
        "is_recommendation": True,
        "show_slack": True,
        "background": "#81c147",
    },
}


def _normalize_display_label(raw: str | None) -> str:
    value = str(raw or "").strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value


_DECISION_MESSAGE_OVERRIDES: dict[str, str] = {
    "BUY": "✅ 신규 매수",
}


DECISION_MESSAGES = {
    key: _normalize_display_label(cfg.get("display_name"))
    for key, cfg in BACKTEST_STATUS_LIST.items()
    if isinstance(cfg, dict) and cfg.get("display_name")
}

for override_key, override_value in _DECISION_MESSAGE_OVERRIDES.items():
    DECISION_MESSAGES[override_key] = override_value

DECISION_MESSAGES = {key: value for key, value in DECISION_MESSAGES.items() if value}

__all__ = [
    "BACKTEST_STATUS_LIST",
    "DECISION_MESSAGES",
]
