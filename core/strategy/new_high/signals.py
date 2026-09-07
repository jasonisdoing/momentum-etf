"""신고가 신호와 판정 기간 — 백테스트·운용 현황·튜닝의 단일 소스."""

from __future__ import annotations

import pandas as pd

# 신고가 판정 창 — 거래일 수가 아니라 달력 기간으로 자른다. 거래일로 고정하면
# 공휴일 수에 따라 실제 기간이 흔들려 이름과 어긋난다(12개월 × 20거래일 = 240거래일은
# 실제 52주보다 짧아서, 1년 전 고점이 창 밖으로 일찍 밀려난다).
HIGH_WINDOW_WEEKS = 52
HIGH_WINDOW = f"{HIGH_WINDOW_WEEKS * 7}D"
# 창 안에 이만큼 거래일이 있어야 판정한다. 364일 창의 거래일 수는 국내 238~249일,
# 미국 249~253일이라 정상 종목은 통과하고, 상장 1년 미만·장기 거래정지만 걸러진다.
HIGH_WINDOW_MIN_DAYS = 230


def compute_signals(panel: dict[str, pd.DataFrame], exit_ma_days: int) -> dict[str, pd.DataFrame]:
    """돌파·이탈 신호와 거래대금 급증 배수(진입 정렬 기준)를 한 번에 만든다."""
    close_df = panel["close"]
    # 진입 판정은 **직전 최고 종가** 기준이다. 종가끼리 비교하므로 종가가 오르는 동안
    # 신호가 끊기지 않는다. 장중 고가와 비교하면 전날 꼬리를 못 넘는 날 신호가 끊겨
    # 상승 중에도 진입 기회를 놓친다(한국 개별주 60개월 기준 그런 날이 34%).
    # 오늘은 창에서 뺀다 — 오늘 종가가 '직전' 최고를 넘었는지를 본다.
    prior_high = close_df.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW_MIN_DAYS).max().shift(1)
    # 관례상의 '52주 신고가'(장중 고가)는 화면 표시용으로만 쓴다 — 판정에는 쓰지 않는다.
    prior_high_intraday = panel["high"].rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW_MIN_DAYS).max().shift(1)
    exit_ma = close_df.rolling(exit_ma_days, min_periods=exit_ma_days).mean()
    value_df = panel["value"]
    from utils.trade_value import trade_value_multiplier_frame

    return {
        # 돌파 판정은 **종가** 기준이다 — 장중에 잠깐 찍고 밀린 것은 돌파로 보지 않는다.
        "breakout": close_df > prior_high,
        "below_ma": close_df < exit_ma,
        # 이탈선 값 자체 — 화면이 "이탈까지 얼마 남았는지"를 보여주는 데 쓴다(판정과 같은 선).
        "exit_ma": exit_ma,
        "prior_high": prior_high,
        "prior_high_intraday": prior_high_intraday,
        # 20일 평균 거래대금 대비 배수 — 돌파에 자금이 실렸는지 본다.
        # **당일을 분모에 포함**한다. 급증한 당일이 스스로 평균을 끌어올려 배수가 눌리고
        # (실제 20배가 10.3배로 표기) 표기값 상한이 20배가 되지만, 참고하는 외부 시스템이
        # 같은 정의를 쓴다(NHN 777억→5.3배, 855억→5.7배 — 분모가 당일 값을 따라 움직인다).
        # 하한 최적값(kor 5배 등)도 이 정의 위에서 찾은 것이라 바꾸면 다시 잡아야 한다.
        "value_mult": trade_value_multiplier_frame(value_df),
    }
