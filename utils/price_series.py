"""가격 시계열 정제 — 백테스트·선정이 같은 규칙으로 값을 읽게 한다."""

from __future__ import annotations

import pandas as pd


def positive_prices(series: pd.Series) -> pd.Series:
    """0 이하 가격을 결측으로 돌린다.

    거래정지 구간에서 0 이 들어온 칸이 있다(국내 개별주 캐시 기준 시가·고가에서 나온다).
    그대로 쓰면 체결가가 0 이 되어 수익률이 -100% 나 무한대가 되고, 신고가 판정도 어긋난다.
    값을 지어내지 않고 없는 것으로 둔다.
    """
    return pd.to_numeric(series, errors="coerce").where(lambda x: x > 0)


def period_return_pct(series: pd.Series, months: int, eval_date: pd.Timestamp) -> float | None:
    """`eval_date` 까지의 마지막 종가를 `months` 개월 전 종가와 비교한 수익률(%).

    기준일이 휴장이면 그 이전 마지막 거래일 종가를 쓴다. 데이터가 모자라면 None.
    화면마다 같은 계산이 흩어져 있어 공용으로 뺀 것이며, 기간 수익률이 필요한
    곳은 이 함수를 쓴다.
    """
    normalized = pd.to_numeric(series, errors="coerce").dropna()
    if normalized.empty:
        return None

    latest_candidates = normalized[normalized.index <= eval_date]
    if latest_candidates.empty:
        return None
    latest_price = float(latest_candidates.iloc[-1])

    target_date = latest_candidates.index[-1] - pd.DateOffset(months=months)
    base_candidates = normalized[normalized.index <= target_date]
    if base_candidates.empty:
        return None
    base_price = float(base_candidates.iloc[-1])
    if base_price <= 0:
        return None

    return round(((latest_price / base_price) - 1.0) * 100.0, 2)
