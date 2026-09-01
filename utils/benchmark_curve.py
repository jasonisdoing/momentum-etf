"""벤치마크 누적 곡선 — **시작일 시가를 1** 로 둔다. 모든 전략 백테스트가 함께 쓴다.

전략은 시작일 **시가**에 체결한다(모멘텀 교체·신고가 진입·포트폴리오 최초 매수 모두 시가다).
그런데 벤치마크를 첫날 **종가** 기준으로 재면, 그날 시가→종가 변동이 벤치마크에서만 빠진다.
같은 날 같은 돈으로 시작한 두 곡선이 아니게 되고, 그 차이가 그대로 「초과(%p)」에 실린다.

실제로 VOO 24개월 구간에서 시가 기준 +41.01% vs 종가 기준 +43.12% 로 2.1%p 가 갈렸다.
시작일 갭이 큰 구간일수록 벌어지므로, 기간을 바꿔 가며 비교할 때 특히 어긋난다.
"""

from __future__ import annotations

import pandas as pd

from utils.price_series import positive_prices


def load_benchmark_frame(pool: str) -> pd.DataFrame:
    """풀 설정의 벤치마크 티커 가격 프레임(Open·Close 포함).

    `momentum_service.load_benchmark_close` 와 같은 캐시를 보되 시가까지 들고 온다.
    """
    from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types
    from utils.momentum_service import _pool_benchmark

    benchmark = _pool_benchmark(pool)
    frame = load_cached_frames_bulk_from_all_ticker_types([benchmark["ticker"]]).get(benchmark["ticker"])
    if frame is None or frame.empty:
        raise RuntimeError(f"벤치마크({benchmark['name']}) 가격 캐시를 불러올 수 없습니다.")
    return frame


def growth_from_frame(frame: pd.DataFrame, index: pd.Index, *, label: str = "벤치마크") -> pd.Series:
    """가격 프레임에서 누적 배수를 만든다 — **첫날 시가를 1** 로 둔다.

    첫날 값은 `첫날 종가 ÷ 첫날 시가`(그날 시가에 사서 종가까지 들고 간 결과)이고,
    이후는 그 시가 대비 종가 비율이다. 전략 곡선과 같은 시점·같은 기준이 된다.

    시가를 못 구하면 **에러**다 — 종가로 슬쩍 대체하면 이 함수를 만든 이유가 사라진다.
    """
    close = positive_prices(frame["Close"]).dropna().reindex(index, method="ffill")
    if close.empty or pd.isna(close.iloc[0]):
        raise RuntimeError(f"{label} 종가가 비어 있습니다.")

    opens = positive_prices(frame["Open"]).dropna() if "Open" in frame.columns else pd.Series(dtype=float)
    first_day = index[0]
    base = opens.get(first_day)
    if base is None or pd.isna(base) or float(base) <= 0:
        raise RuntimeError(f"{label} {first_day.date()} 시가가 없어 시작 기준을 잡을 수 없습니다.")

    return close / float(base)


def benchmark_growth(pool: str, index: pd.Index) -> pd.Series:
    """종목풀 설정 벤치마크의 누적 배수 (첫날 시가 = 1)."""
    return growth_from_frame(load_benchmark_frame(pool), index)
