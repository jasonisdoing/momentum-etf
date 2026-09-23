"""
이동평균 계산 유틸리티.

종류를 고르는 자리가 **둘**이다. 섞지 않는다.

- **종목풀 판정**(추세선·이격도·순위·신고가 이탈선) → 그 풀의 `pool_settings.MOVING_AVERAGE_TYPE`.
  이평선 **일수**가 이미 풀별이라 종류도 같은 자리에 둔다. 실측에서 최적이 풀마다 갈렸다
  (60개월: us_stock SMA +828.6% vs EMA +608.1%, kor_stock SMA +1393.4% vs EMA +1958.5%).
  → `pool_moving_average_type(pool)`
- **풀에 속하지 않는 계산**(시장지수 추세·레짐, 레버리지) → `config.MOVING_AVERAGE_TYPE`.
  지수는 어느 풀 것도 아니라 고를 근거가 없다. → `get_moving_average_type()`
"""

import pandas as pd


def _normalize(value: object, source: str) -> str:
    text = str(value or "").strip().upper()
    if text not in {"SMA", "EMA"}:
        raise ValueError(f"{source} 는 'SMA' 또는 'EMA' 여야 합니다: {value!r}")
    return text


def get_moving_average_type() -> str:
    """**공통** 이동평균 종류 — 시장지수 추세·레버리지용. 종목풀 판정에는 쓰지 않는다."""
    from config import MOVING_AVERAGE_TYPE

    return _normalize(MOVING_AVERAGE_TYPE, "MOVING_AVERAGE_TYPE")


def pool_moving_average_type(pool: str) -> str:
    """그 **종목풀**의 이동평균 종류 — 전략 판정이 쓰는 값.

    이평선 일수와 같은 필수 항목이다. 미설정이면 조용히 기본값을 쓰지 않고 에러를 낸다
    — 종목풀 설정에서 명시적으로 저장해야 한다.
    """
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    if "MOVING_AVERAGE_TYPE" not in settings:
        raise ValueError(f"'{pool}' 종목풀 설정에 MOVING_AVERAGE_TYPE 이 없습니다 — 종목풀 설정에서 저장하세요.")
    return _normalize(settings["MOVING_AVERAGE_TYPE"], f"'{pool}' 의 MOVING_AVERAGE_TYPE")


def calculate_sma(prices: pd.Series, period: int, min_periods: int = 1) -> pd.Series:
    """단순 이동평균(SMA)."""
    return prices.rolling(window=period, min_periods=min_periods).mean()


def calculate_ema(prices: pd.Series, period: int, min_periods: int = 1) -> pd.Series:
    """지수 이동평균(EMA). span=period(=SMA와 같은 '기간'), 초기 편향 최소화(adjust=False)."""
    return prices.ewm(span=period, adjust=False, min_periods=min_periods).mean()


def calculate_moving_average(
    prices: pd.Series, period: int, min_periods: int = 1, *, ma_type: str | None = None
) -> pd.Series:
    """SMA 또는 EMA 이동평균을 계산한다.

    Args:
        prices: 가격 시리즈(또는 [일자 × 티커] 프레임)
        period: 이동평균 기간
        min_periods: 최소 표본 수(기본 1 = 첫 봉부터 부분평균). 기존 rolling 기본과 맞추려면 period 를 넘긴다.
        ma_type: "SMA"/"EMA". **전략 판정은 그 풀의 값**(`pool_moving_average_type`)을 넘긴다.
            생략하면 공통값(`config.MOVING_AVERAGE_TYPE`) — 시장지수·레버리지처럼 풀이 없는 계산용이다.
    """
    resolved = _normalize(ma_type, "ma_type") if ma_type is not None else get_moving_average_type()
    if resolved == "EMA":
        return calculate_ema(prices, period, min_periods=min_periods)
    return calculate_sma(prices, period, min_periods=min_periods)


__all__ = [
    "calculate_ema",
    "calculate_moving_average",
    "calculate_sma",
    "get_moving_average_type",
    "pool_moving_average_type",
]
