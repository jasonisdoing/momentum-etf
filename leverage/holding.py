"""레버리지 보유 기간(보유시작일·보유일수) 공통 로직.

보유일수는 매번 백테스트로 추정하지 않고, **확정된 보유시작일(holding_start_date)** 에서
오늘(장중 포함)까지의 거래일수를 센다. 보유시작일은 포지션이 실제로 바뀐 날에만 새로
기록하고, 포지션이 유지되면 기존 값을 그대로 유지한다(임의 보정·리셋 없음).
"""

from __future__ import annotations


def resolve_holding_start_date(
    prev_target: str | None,
    prev_start_date: str | None,
    new_target: str,
    confirmed_date: str,
) -> str:
    """확정된 보유시작일을 결정한다.

    - 포지션 유지(prev_target == new_target) + 기존 시작일 존재 → **기존 시작일 유지**.
    - 포지션 변경 또는 최초 기록 → 확정일(confirmed_date)을 시작일로 새로 잡는다.
    """
    if prev_target == new_target and prev_start_date:
        return prev_start_date
    return confirmed_date


def holding_period_info(target: str, start_date_str: str) -> tuple[int, float | None]:
    """보유시작일(start_date_str) 기준 (보유 거래일수, 보유시작일 종가)를 한 번의 조회로 반환한다.

    - 보유 거래일수: start_date_str 부터 오늘(장중 포함)까지의 거래 영업일수.
    - 보유시작일 종가: 누적 수익률 기준가(현재가 / 이 값 - 1). 조회 실패 시 None.
    """
    import pandas as pd

    from utils.data_loader import fetch_ohlcv, is_trading_day

    ticker = "237350" if target == "CASH" or not target else target
    country = "kor"

    try:
        df = fetch_ohlcv(ticker, country, months_back=None, date_range=[start_date_str, None], ticker_type="etf")
        if df is not None and not df.empty:
            count = len(df)
            today_ts = pd.Timestamp.today().normalize()
            if is_trading_day(country, today_ts):
                df_last_day = pd.Timestamp(df.index[-1]).normalize()
                if today_ts > df_last_day:
                    count += 1
            start_close = float(df["Close"].iloc[0])
            return count, start_close
    except Exception as e:
        print(f"[holding_period_info] fetch_ohlcv 실패 ({ticker}, {start_date_str}): {e}")

    try:
        start_date = pd.Timestamp(start_date_str).normalize()
        today = pd.Timestamp.today().normalize()
        return len(pd.bdate_range(start_date, today)), None
    except Exception:
        return 0, None


def count_holding_trading_days(target: str, start_date_str: str) -> int:
    """start_date_str 부터 오늘(장중 포함)까지의 거래 영업일수."""
    return holding_period_info(target, start_date_str)[0]
