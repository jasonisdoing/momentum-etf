"""MAPS 전략 의사결정 평가 모듈"""

import pandas as pd

from strategies.maps.constants import DECISION_MESSAGES, DECISION_NOTES


def _format_trend_break_phrase(ma_value: float | None, price_value: float | None, ma_period: int | None) -> str:
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


def _calc_days_left(block_info: dict | None, cooldown_days: int | None) -> int | None:
    if not cooldown_days or cooldown_days <= 0 or not isinstance(block_info, dict):
        return None
    try:
        return max(cooldown_days - int(block_info.get("days_since", 0)) + 1, 0)
    except (TypeError, ValueError):
        return None


def _format_cooldown_message(days_left: int | None, action: str = "") -> str:
    if days_left is None:
        template = DECISION_NOTES.get("COOLDOWN_GENERIC", "쿨다운 {days}일 대기중")
        return template.replace("{days}", "?")

    if days_left > 0:
        template = DECISION_NOTES.get("COOLDOWN_GENERIC", "쿨다운 {days}일 대기중")
        base = template.replace("{days}", str(days_left))
        return f"{base} ({action})" if action else base
    else:
        template = DECISION_NOTES.get("COOLDOWN_GENERIC", "쿨다운 {days}일 대기중")
        return template.replace("{days}", "0")


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
        ma_period: int,
        score: float,
        rsi_score: float,
        stop_loss_threshold: float | None,
        rsi_sell_threshold: float,
        sell_cooldown_info: dict | None,
        cooldown_days: int,
    ) -> tuple[str, str]:
        """
        매도 여부를 판단합니다.

        Returns:
            (new_state, phrase)
        """
        phrase = ""
        new_state = current_state

        if current_state not in ("HOLD",):
            return current_state, phrase

        hold_ret = (price / avg_cost - 1.0) * 100.0 if avg_cost > 0 else 0.0

        if stop_loss_threshold is not None and hold_ret <= float(stop_loss_threshold):
            new_state = "CUT_STOPLOSS"
            phrase = DECISION_MESSAGES.get("CUT_STOPLOSS", "손절매도")
        elif rsi_sell_threshold is not None and rsi_score >= rsi_sell_threshold:
            new_state = "SELL_RSI"
            phrase = f"RSI 과매수 (RSI점수: {rsi_score:.1f})"
        elif score <= 0:
            new_state = "SELL_TREND"
            phrase = _format_trend_break_phrase(ma_value, price, ma_period)

        # 쿨다운 중이면 일반 매도(추세, RSI)는 HOLD로 유지
        # 손절은 쿨다운 무시
        if sell_cooldown_info and new_state in ("SELL_RSI", "SELL_TREND"):
            days_left = _calc_days_left(sell_cooldown_info, cooldown_days)
            if days_left and days_left > 0:
                new_state = "HOLD"
                action = f"{days_left}일 후 매도 가능"
                phrase = _format_cooldown_message(days_left, action)

        return new_state, phrase

    def check_buy_signal(
        self,
        score: float,
        buy_cooldown_info: dict | None,
        cooldown_days: int,
    ) -> tuple[bool, str]:
        """
        매수 시그널 발생 여부를 확인합니다.

        Returns:
            (is_buy_signal, phrase)
        """
        from logic.backtest.signals import has_buy_signal

        if has_buy_signal(score, 0):
            if buy_cooldown_info:
                days_left = _calc_days_left(buy_cooldown_info, cooldown_days)
                if days_left and days_left > 0:
                    action = f"{days_left}일 후 매수 가능"
                    phrase = _format_cooldown_message(days_left, action)
                    return False, phrase
                else:
                    return True, "쿨다운 대기중(오늘 매수 가능)"
            return True, DECISION_MESSAGES.get("NEW_BUY", "")

        # 점수 미달 메시지
        if pd.isna(score):
            return False, "추세 이탈 (점수 없음)"
        return False, f"추세 이탈 (현재 {score:.1f})"
