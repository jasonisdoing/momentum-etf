"""가격 계산 관련 공통 함수."""

import pandas as pd

from config import BACKTEST_SLIPPAGE


def calculate_trade_price(
    current_index: int,
    total_days: int,
    open_values: any,
    close_values: any,
    country_code: str,
    is_buy: bool,
) -> float:
    """
    거래 가격 계산: 다음날 시초가 + 슬리피지

    Args:
        current_index: 현재 인덱스 (i)
        total_days: 전체 거래일 수
        open_values: Open 가격 배열
        close_values: Close 가격 배열
        country_code: 국가 코드
        is_buy: 매수 여부 (True: 매수, False: 매도)

    Returns:
        거래 가격
    """
    # 다음날 시초가 사용
    if current_index + 1 < total_days:
        next_open = open_values[current_index + 1]
        if pd.notna(next_open):
            base_price = float(next_open)
        else:
            # 다음날 시초가가 없으면 당일 종가 사용
            base_price = float(close_values[current_index]) if pd.notna(close_values[current_index]) else 0.0
    else:
        # 마지막 날은 당일 종가 사용
        base_price = float(close_values[current_index]) if pd.notna(close_values[current_index]) else 0.0

    if base_price <= 0:
        return 0.0

    # 슬리피지 적용
    slippage_config = BACKTEST_SLIPPAGE.get(country_code, BACKTEST_SLIPPAGE.get("kor", {}))

    if is_buy:
        # 매수: 시초가보다 높은 가격
        slippage_pct = slippage_config.get("buy_pct")
        trade_price = base_price * (1 + slippage_pct / 100)
    else:
        # 매도: 시초가보다 낮은 가격
        slippage_pct = slippage_config.get("sell_pct")
        trade_price = base_price * (1 - slippage_pct / 100)

    return trade_price


def resolve_highest_price_since_buy(series: any, buy_date: any) -> float | None:
    """매수일 이후 최고가를 반환합니다."""
    if buy_date is None:
        return None

    if not isinstance(series, pd.Series) or series.empty:
        return None

    try:
        buy_ts = pd.to_datetime(buy_date).normalize()
    except Exception:
        return None

    cleaned = series.dropna().copy()
    if cleaned.empty:
        return None

    try:
        cleaned.index = pd.to_datetime(cleaned.index).normalize()
    except Exception:
        return None

    future_slice = cleaned.loc[cleaned.index >= buy_ts]
    if future_slice.empty:
        # 미래 데이터가 없으면 데이터의 마지막 값을 반환 (보수적 접근)
        val = cleaned.iloc[-1]
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    try:
        return float(future_slice.max())
    except (TypeError, ValueError):
        return None


def resolve_entry_price(series: any, buy_date: any) -> float | None:
    """매수일 이후 첫 종가를 반환합니다 (실제 진입가 추정)."""
    if buy_date is None:
        return None

    if not isinstance(series, pd.Series) or series.empty:
        return None

    try:
        buy_ts = pd.to_datetime(buy_date).normalize()
    except Exception:
        return None

    cleaned = series.dropna().copy()
    if cleaned.empty:
        return None

    try:
        cleaned.index = pd.to_datetime(cleaned.index).normalize()
    except Exception:
        return None

    future_slice = cleaned.loc[cleaned.index >= buy_ts]
    if future_slice.empty:
        # 미래 데이터가 없으면 데이터의 마지막 값을 반환
        val = cleaned.iloc[-1]
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    entry_val = future_slice.iloc[0]
    try:
        return float(entry_val)
    except (TypeError, ValueError):
        return None
