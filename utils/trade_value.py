"""거래대금 배수 계산과 한국 장중 환산 규칙의 단일 소스.

배치 최신값·신고가 백테스트가 같은 계산을 써야 화면과 과거 재현이 갈리지 않는다.
거래대금은 ``종가 × 거래량``, 배수는 당일을 포함한 최근 20개 **유효 관측값**의
평균 대비다. 특정 날짜의 거래량이 비어도 그 뒤 20거래일을 통째로 비우지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from config import MARKET_SCHEDULES

TRADE_VALUE_WINDOW = 20


def trade_value_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """종가·거래량이 모두 있는 날짜의 거래대금 시계열."""
    close_values = pd.to_numeric(close, errors="coerce").where(lambda values: values > 0)
    volume_values = pd.to_numeric(volume, errors="coerce")
    return close_values * volume_values


def trade_value_multiplier_series(values: pd.Series) -> pd.Series:
    """최근 20개 유효 거래대금 평균 대비 배수. 반환 인덱스는 입력과 같다."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    multiplier = valid / valid.rolling(TRADE_VALUE_WINDOW, min_periods=TRADE_VALUE_WINDOW).mean()
    return multiplier.reindex(numeric.index)


def trade_value_multiplier_frame(values: pd.DataFrame) -> pd.DataFrame:
    """티커별로 유효 관측값을 따로 세어 거래대금 배수를 계산한다."""
    return pd.DataFrame(
        {ticker: trade_value_multiplier_series(values[ticker]) for ticker in values.columns},
        index=values.index,
    )


def latest_trade_value_fields(close: pd.Series, volume: pd.Series) -> dict[str, float] | None:
    """배치가 ``stock_meta``에 저장할 최신 거래대금·배수·직전 19일 합."""
    values = trade_value_series(close, volume).dropna()
    if len(values) < TRADE_VALUE_WINDOW:
        return None
    window = values.iloc[-TRADE_VALUE_WINDOW:]
    base = float(window.mean())
    if base <= 0:
        return None
    return {
        "trade_value": round(float(values.iloc[-1]), 2),
        "trade_value_mult": round(float(values.iloc[-1]) / base, 4),
        # 다음 거래일 장중 계산의 '직전 19거래일' — 최신 완료일을 포함하고 가장 오래된 날을 뺀다.
        "trade_value_sum19": round(float(window.iloc[1:].sum()), 2),
    }


def session_elapsed_fraction(country: str) -> float | None:
    """그 시장의 프리마켓 시작~애프터 종료 경과 비율(시작 전 0, 종료 후 1).

    시간표를 모르는 시장은 None — 비율을 지어내지 않는다. 누적 거래대금을 하루 기준으로
    환산할 때 쓰므로, 정규장만이 아니라 체결이 일어나는 구간 전체를 분모로 본다.
    """
    schedule = (MARKET_SCHEDULES or {}).get(str(country).strip().lower())
    if not isinstance(schedule, dict):
        return None
    start, end = schedule.get("premarket_open"), schedule.get("aftermarket_close")
    tz_name = str(schedule.get("timezone") or "").strip()
    if start is None or end is None or not tz_name:
        return None
    now = datetime.now(ZoneInfo(tz_name))
    open_at = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    close_at = now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    span = (close_at - open_at).total_seconds()
    if span <= 0:
        return None
    return min(max((now - open_at).total_seconds() / span, 0.0), 1.0)


def session_live_fraction(country: str) -> float | None:
    """거래 세션이 진행 중인 거래일에만 경과 비율을 반환한다."""
    fraction = session_elapsed_fraction(country)
    if fraction is None or not 0.0 < fraction < 1.0:
        return None
    from utils.trading_calendar import is_trading_day

    return fraction if is_trading_day(country) else None


def live_min_value_mult(min_value_mult: float | None, country: str) -> float | None:
    """장중 누적 거래대금에 적용할 시간 비례 하한.

    누적 배수는 장이 진행될수록 커지므로 하한도 같은 비율로 낮춰야 공평하다(개장 직후
    2배를 요구하면 아무것도 통과하지 못한다). 장중이 아니면 하한을 그대로 돌려준다.
    """
    if min_value_mult is None:
        return None
    fraction = session_live_fraction(country)
    return float(min_value_mult) * fraction if fraction is not None else float(min_value_mult)


__all__ = [
    "TRADE_VALUE_WINDOW",
    "session_elapsed_fraction",
    "session_live_fraction",
    "latest_trade_value_fields",
    "live_min_value_mult",
    "trade_value_multiplier_frame",
    "trade_value_multiplier_series",
    "trade_value_series",
]
