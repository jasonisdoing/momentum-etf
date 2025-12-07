"""노트/메시지 생성 관련 공통 함수."""

import pandas as pd

from strategies.maps.constants import DECISION_NOTES


def format_trend_break_phrase(
    ma_value: float | None,
    price_value: float | None,
    ma_period: int | None,
) -> str:
    """추세 이탈 메시지 포맷팅."""
    if ma_value is None or pd.isna(ma_value) or price_value is None or pd.isna(price_value):
        threshold = ma_value if (ma_value is not None and not pd.isna(ma_value)) else 0.0
        return f"{DECISION_NOTES['TREND_BREAK']}({threshold:,.0f}원 이하)"

    diff = ma_value - price_value
    direction = "낮습니다" if diff >= 0 else "높습니다"
    period_text = ""
    if ma_period:
        try:
            period_text = f"{int(ma_period)}일 "
        except (TypeError, ValueError):
            period_text = ""
    return (
        f"{DECISION_NOTES['TREND_BREAK']}"
        f"({period_text}평균 가격 {ma_value:,.0f}원 보다 {abs(diff):,.0f}원 {direction}.)"
    )


def format_min_score_phrase(score_value: float | None, min_buy_score: float) -> str:
    """최소 점수 미달 메시지 포맷팅."""
    template = DECISION_NOTES.get("MIN_SCORE", "최소 {min_buy_score:.1f}점수 미만")
    try:
        base = template.format(min_buy_score=min_buy_score)
    except Exception:
        base = f"최소 {min_buy_score:.1f}점수 미만"

    if score_value is None or pd.isna(score_value):
        return f"{base} (현재 점수 없음)"
    return f"{base} (현재 {score_value:.1f})"
