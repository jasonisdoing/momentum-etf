"""모멘텀 신호 계산 — 백테스트·운용 현황·튜닝이 공유한다."""

from __future__ import annotations

import pandas as pd

from core.strategy.scoring import calculate_maps_score, hold_eligible, rank_score
from utils.moving_averages import calculate_moving_average

# 진입 문턱의 변동성 창(거래일) — 일간 수익률 표준편차(%). 문턱 = 배수 × 이 값.
ENTRY_VOL_WINDOW = 20


def daily_volatility_pct(close):
    """최근 20일 일간 수익률 표준편차(%) — "하루에 보통 이만큼 출렁인다".

    진입 문턱 판정(`entry_signal`)과 화면 변동성 컬럼(순위·모멘텀·신고가)이 **같은 이 값**을
    쓴다 — 화면 숫자와 판정 기준이 갈리면 안 된다. Series·DataFrame 모두 받는다.
    """
    return close.pct_change().rolling(ENTRY_VOL_WINDOW).std() * 100


def entry_signal(
    close_df: pd.DataFrame, signals: dict[str, pd.DataFrame], entry_vol_mult: float | None
) -> pd.DataFrame:
    """진입 자격 — 보유 자격(`eligible`)에 변동성 문턱을 얹는다. **청산 판정은 불변.**

    문턱 = ``entry_vol_mult`` × 그 종목의 20일 일간 변동성(%). 단기·장기 이격이 모두
    문턱 이상이어야 진입한다 — 청산선(0선) 바로 위의 종목을 사서 하루 만에 되파는 왕복을
    막는다. 진입·청산 기준선이 달라지는 히스테리시스라, 보유 유지는 기존 0선 그대로다.
    None 이면 문턱 없음(진입 = 보유 자격). 변동성을 못 잰 종목(워밍업)은 진입 불가 —
    값을 추정하지 않는다.
    """
    if entry_vol_mult is None:
        return signals["eligible"]
    volatility = daily_volatility_pct(close_df)
    floor = float(entry_vol_mult) * volatility
    return signals["eligible"] & (signals["long"] > floor) & (signals["short"] >= floor) & volatility.notna()


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
