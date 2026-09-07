"""조회한 가격 프레임을 전략 공통 패널로 변환한다. 외부 데이터 조회는 하지 않는다."""

from __future__ import annotations

import pandas as pd

from utils.price_series import positive_prices as _positive


def build_price_panel(universe: list[dict[str, str]], frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """(날짜 × 티커) 종가·시가·고가·거래대금 표. 백테스트와 선정이 같은 값을 쓴다."""
    from utils.trade_value import trade_value_series

    closes, opens, highs, values = {}, {}, {}, {}
    for row in universe:
        ticker = row["ticker"]
        frame = frames.get(ticker)
        if frame is None or frame.empty or "Close" not in frame or "Open" not in frame or "High" not in frame:
            continue
        close = _positive(frame["Close"])
        volume = pd.to_numeric(frame["Volume"], errors="coerce") if "Volume" in frame else None
        value_close_column = next(
            (column for column in ("unadjusted_close", "Close", "close") if column in frame.columns), None
        )
        closes[ticker] = close
        opens[ticker] = _positive(frame["Open"])
        highs[ticker] = _positive(frame["High"])
        values[ticker] = (
            trade_value_series(frame[value_close_column], volume)
            if volume is not None and value_close_column is not None
            else pd.Series(index=frame.index, dtype=float)
        )

    if not closes:
        raise RuntimeError("가격 캐시를 불러오지 못해 신고가를 판정할 수 없습니다.")

    close_df = pd.DataFrame(closes).sort_index()
    # 종가가 **한 종목도 없는 날**은 거래일로 치지 않는다. 가격 캐시가 거래량만 채우고
    # OHLC 는 비운 행을 만들 때가 있는데(us_etf 2026-08-28), 그걸 마지막 거래일로 삼으면
    # 백테스트가 그날 보유 평가액을 0 으로 매겨 하루 만에 -100% 로 무너진다.
    close_df = close_df.dropna(axis=0, how="all")
    if close_df.empty:
        raise RuntimeError("가격 캐시에 종가가 있는 거래일이 없습니다.")
    return {
        "close": close_df,
        "open": pd.DataFrame(opens).reindex(close_df.index),
        "high": pd.DataFrame(highs).reindex(close_df.index),
        "value": pd.DataFrame(values).reindex(close_df.index),
    }
