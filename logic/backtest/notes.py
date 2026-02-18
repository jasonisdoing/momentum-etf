"""노트/메시지 생성 관련 공통 함수."""

import pandas as pd

from strategies.maps.constants import DECISION_NOTES


def format_trend_break_phrase(
    ma_value: float | None,
    price_value: float | None,
    ma_days: int | None,
) -> str:
    """추세 이탈 메시지 포맷팅."""
    if ma_value is None or pd.isna(ma_value) or price_value is None or pd.isna(price_value):
        threshold = ma_value if (ma_value is not None and not pd.isna(ma_value)) else 0.0
        return f"{DECISION_NOTES['TREND_BREAK']}({threshold:,.0f}원 이하)"

    diff = ma_value - price_value
    direction = "낮습니다" if diff >= 0 else "높습니다"
    period_text = ""
    if ma_days:
        try:
            period_text = f"{int(ma_days)}일 "
        except (TypeError, ValueError):
            period_text = ""
    return (
        f"{DECISION_NOTES['TREND_BREAK']}"
        f"({period_text}평균 가격 {ma_value:,.0f}원 보다 {abs(diff):,.0f}원 {direction}.)"
    )
