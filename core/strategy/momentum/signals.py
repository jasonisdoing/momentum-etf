"""모멘텀 신호 계산 — 백테스트·운용 현황·튜닝이 공유한다."""

from __future__ import annotations

import pandas as pd

from core.strategy.scoring import calculate_maps_score, hold_eligible, rank_score
from utils.moving_averages import calculate_moving_average


def compute_signals(panel: dict[str, pd.DataFrame], short_ma_days: int, long_ma_days: int) -> dict[str, pd.DataFrame]:
    """이평선 두 개로 만드는 신호 표(행 = 거래일, 열 = 종목) — 백테스트·화면이 같은 값을 본다.

    ``short``/``long`` 은 이격률(%)이다. 이평선을 못 채운 날은 NaN 이고, 그런 날은 ``known``
    이 거짓이라 사고팔지 않는다 — 값을 추정하지 않는다.
    """
    close_df = panel["close"]
    short_gap, long_gap, ready = {}, {}, {}
    for ticker in close_df.columns:
        close = close_df[ticker]
        short_ma = calculate_moving_average(close, short_ma_days, min_periods=short_ma_days)
        long_ma = calculate_moving_average(close, long_ma_days, min_periods=long_ma_days)
        short_gap[ticker] = calculate_maps_score(close, short_ma)
        long_gap[ticker] = calculate_maps_score(close, long_ma)
        # 판정 가능 여부는 **이평선 자체**로 본다 — `calculate_maps_score` 는 못 채운 날을
        # 0 으로 메우므로(fillna) 이격만 보면 워밍업 구간이 '이격 0' 인 정상 값처럼 보인다.
        ready[ticker] = close.notna() & short_ma.notna() & long_ma.notna()
    short_frame = pd.DataFrame(short_gap, index=close_df.index)
    long_frame = pd.DataFrame(long_gap, index=close_df.index)
    known = pd.DataFrame(ready, index=close_df.index)
    # 보유 자격은 순위 화면·종목풀 백테스트와 **같은 공용 규칙**(`hold_eligible`)이다.
    eligible = known & hold_eligible(long_frame, short_frame)
    return {
        "short": short_frame,
        "long": long_frame,
        "known": known,
        "eligible": eligible,
        # 청산은 '자격을 잃었다고 판정할 수 있는 날'만 — 데이터가 없으면 유지한다.
        "exit": known & ~eligible,
        # 진입 우선순위 — 순위 화면과 같은 단일 기준(`rank_score`, 정의는 장기 이격률).
        "priority": rank_score(long_frame, short_frame),
    }
