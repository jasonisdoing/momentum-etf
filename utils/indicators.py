"""
보조지표 계산 함수 모음 (e.g., SuperTrend, EMA, etc.)
"""
import pandas as pd

def supertrend_direction(df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
    """SuperTrend 방향을 계산합니다. (+1: 상향, -1: 하향)"""
    high = df["High"]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]

    low = df["Low"]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    hl2 = (high + low) / 2.0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = -1
            continue
        prev_st = st.iloc[i-1]
        prev_dir = direction.iloc[i-1]
        curr_close = close.iloc[i]

        if curr_close > prev_st:
            direction.iloc[i] = 1
        elif curr_close < prev_st:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir

        if direction.iloc[i] == 1:
            st.iloc[i] = max(lowerband.iloc[i], prev_st if prev_dir == 1 else lowerband.iloc[i])
        else:
            st.iloc[i] = min(upperband.iloc[i], prev_st if prev_dir == -1 else upperband.iloc[i])

    return direction.fillna(-1).astype(int)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR)를 계산합니다.
    """
    if not all(col in df.columns for col in ["High", "Low", "Close"]):
        return pd.Series(dtype=float, index=df.index)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR은 일반적으로 지수이동평균(EMA)을 사용합니다.
    atr_series = tr.ewm(com=period - 1, min_periods=period).mean()
    return atr_series