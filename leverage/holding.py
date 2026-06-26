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


def count_holding_trading_days(target: str, start_date_str: str) -> int:
    """start_date_str(YYYY-MM-DD)부터 오늘(장중 포함)까지의 거래 영업일수를 계산한다."""
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
            return count
    except Exception as e:
        print(f"[count_holding_trading_days] fetch_ohlcv 실패 ({ticker}, {start_date_str}): {e}")

    try:
        start_date = pd.Timestamp(start_date_str).normalize()
        today = pd.Timestamp.today().normalize()
        return len(pd.bdate_range(start_date, today))
    except Exception:
        return 0
