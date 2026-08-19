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
