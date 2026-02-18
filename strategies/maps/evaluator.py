"""MAPS 전략 의사결정 평가 모듈"""

import pandas as pd

from strategies.maps.constants import DECISION_MESSAGES, DECISION_NOTES


def _format_trend_break_phrase(ma_value: float | None, price_value: float | None, ma_days: int | None) -> str:
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


class StrategyEvaluator:
    """MAPS 전략의 매수/매도 의사결정을 담당하는 클래스"""

    def __init__(self):
        pass

    def evaluate_sell_decision(
        self,
        current_state: str,
        price: float,
        avg_cost: float,
        highest_price: float,
        ma_value: float,
        ma_days: int,
        score: float,
    ) -> tuple[str, str]:
        """
        매도 여부를 판단합니다.

        Returns:
            (new_state, phrase)
        """
        phrase = ""
        new_state = current_state

        if score <= 0:
            # 전략 변경: 추세 이탈 시에도 매도하지 않음 (리밸런싱 날에만 교체)
            # new_state = "SELL_TREND"
            # phrase = _format_trend_break_phrase(ma_value, price, ma_days)
            pass

        return new_state, phrase

    def check_buy_signal(
        self,
        score: float,
    ) -> tuple[bool, str]:
        """
        매수 시그널 발생 여부를 확인합니다.

        Returns:
            (is_buy_signal, phrase)
        """
        from core.backtest.signals import has_buy_signal

        if has_buy_signal(score, 0):
            return True, DECISION_MESSAGES.get("NEW_BUY", "")

        # 점수 미달 메시지
        if pd.isna(score):
            return False, "추세 이탈 (점수 없음)"
        return False, f"추세 이탈 (현재 {score:.1f})"
