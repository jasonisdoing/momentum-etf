"""합성 재생 핵심 — 슬리브 곡선·주식 비율 프레임을 받아 월초 재배분으로 굴린다.

계좌·설정·슬리피지 조회는 하지 않는다 — 호출자(`utils/strategy_mix_service`)가 수집해
넘긴다. 합성 백테스트와 운용 배분(슬리브 몫 역산)이 이 한 함수를 공유한다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.strategy.mix.rebalance import rebalance_sleeves


def replay_mix(
    frame: pd.DataFrame,
    stock: pd.DataFrame,
    weights: dict[str, float],
    slippage: dict[str, tuple[float, float]],
    *,
    through_date: str | None,
) -> dict[str, Any]:
    """월초 재배분 재생 — 슬리브 금액·유보 현금의 경로와 합성 곡선을 낸다.

    ``frame`` 은 슬리브별 누적 배수(행 = 날짜 문자열), ``stock`` 은 같은 인덱스의
    슬리브별 주식 비율(0~1)이다. ``through_date`` 가 다음 달이면 아직 월초 종가가 없는
    상태 — 직전 확정 가격으로 같은 재배분을 미리 계산한다(실제 거래일 계산과 같은 함수).
    """
    if len(frame) < 2 or stock.isna().any().any():
        raise RuntimeError("합성에 필요한 공통 가격·현금 비중 데이터가 부족합니다.")
    if through_date is not None and through_date > frame.index[-1] and through_date[:7] != frame.index[-1][:7]:
        # 아직 월초 종가가 없으면 직전 확정 가격으로 목표만 계산한다.
        frame = frame.copy()
        stock = stock.copy()
        frame.loc[through_date] = frame.iloc[-1]
        stock.loc[through_date] = stock.iloc[-1]
    days = list(frame.index)
    values = {key: weights[key] for key in frame.columns}
    cash = weights["cash"]
    curve = {days[0]: sum(values.values()) + cash}
    growth = frame / frame.shift(1)
    for i, day in enumerate(days[1:], start=1):
        for key in values:
            values[key] *= float(growth.at[day, key])
        if day[:7] != days[i - 1][:7]:
            values, cash, _ = rebalance_sleeves(values, cash, weights, stock.loc[day].to_dict(), slippage)
        curve[day] = sum(values.values()) + cash
    return {"curve": pd.Series(curve).sort_index(), "values": values, "cash": cash}
