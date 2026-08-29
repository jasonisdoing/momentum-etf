"""보유 종목 차트 데이터 — 전략 화면(신고가·모멘텀·합성)의 「차트」 탭 공용.

일봉 캔들과 이동평균선(들)을 만들어 준다. 어떤 선을 그릴지는 전략이 정한다 —
신고가는 이탈 이평선 1개, 모멘텀은 단기·장기 2개를 넘긴다. 판정은 하지 않는다.
진입 시점 표시는 화면이 이미 들고 있는 보유 정보로 찍는다.

가격 캐시 원본 프레임을 읽는 것은 저가(Low)가 필요해서다 — 전략 패널은 판정에 쓰는
컬럼만 담는다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import HOLDING_CHART_MONTHS
from utils.price_series import positive_prices as _positive

_CANDLE_KEYS = ("Open", "High", "Low", "Close")


def holding_charts(
    pool: str,
    tickers: list[str],
    ma_days_list: list[int],
    as_of: str | None = None,
    months: int | None = None,
) -> list[dict[str, Any]]:
    """티커별 {ticker, name, candles, ma_lines} 목록. 순서는 넘긴 티커 순서.

    ``months`` 를 안 주면 `config.HOLDING_CHART_MONTHS` — 화면 문구도 같은 값을 받아 쓴다.
    """
    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types
    from utils.portfolio_io import average_buy_price_by_ticker
    from utils.settings_loader import get_ticker_type_settings
    from utils.stock_list_io import _load_ticker_type_stocks_raw

    months = int(HOLDING_CHART_MONTHS if months is None else months)
    wanted = [ticker for ticker in dict.fromkeys(str(t).strip() for t in tickers) if ticker]
    ma_days_list = sorted({int(days) for days in ma_days_list if int(days) > 0})
    if not wanted or not ma_days_list:
        return []
    name_by = {
        str(item.get("ticker") or "").strip(): str(item.get("name") or "").strip()
        for item in _load_ticker_type_stocks_raw(pool)
    }
    frames = load_cached_frames_bulk_from_ticker_types([pool], wanted)
    cutoff = pd.Timestamp(as_of) if as_of else None
    # 내 평균 매입가 — 전 계좌 합산. 같은 티커가 다른 시장에도 있으면(IOO) 이 풀의 통화만 센다.
    pool_currency = str((get_ticker_type_settings(pool) or {}).get("currency") or "").strip()
    avg_buy_by = average_buy_price_by_ticker(wanted, currency=pool_currency or None)

    charts: list[dict[str, Any]] = []
    for ticker in wanted:
        frame = frames.get(ticker)
        if frame is None or frame.empty or any(key not in frame for key in _CANDLE_KEYS):
            continue
        if cutoff is not None:
            frame = frame[frame.index <= cutoff]
        if frame.empty:
            continue
        cols = {key: _positive(frame[key]) for key in _CANDLE_KEYS}
        close = cols["Close"]
        # 화면이 보는 구간만 잘라 보내되, 이평선은 잘린 앞부분까지 써서 계산한다.
        ma_by_days = {days: close.rolling(days, min_periods=days).mean() for days in ma_days_list}
        span = frame.index[frame.index >= frame.index[-1] - pd.DateOffset(months=months)]

        candles: list[dict[str, Any]] = []
        points_by_days: dict[int, list[dict[str, Any]]] = {days: [] for days in ma_days_list}
        for day in span:
            values = [cols[key].get(day) for key in _CANDLE_KEYS]
            if any(pd.isna(value) for value in values):
                continue
            date = str(day.date())
            candles.append(dict(zip(("open", "high", "low", "close"), (float(v) for v in values)), time=date))
            for days, ma in ma_by_days.items():
                if pd.notna(ma.get(day)):
                    points_by_days[days].append({"time": date, "value": float(ma[day])})
        if not candles:
            continue
        charts.append(
            {
                "ticker": ticker,
                "name": name_by.get(ticker) or ticker,
                "candles": candles,
                "ma_lines": [{"ma_days": days, "points": points_by_days[days]} for days in ma_days_list],
                # 내 평균 매입가 — 실제로 들고 있는 종목에만 붙는다(`/ticker` 상세와 같은 값).
                "avg_buy_price": avg_buy_by.get(ticker),
                # 통화 — 화면이 가격에 기호를 붙인다(원 · $ · A$). 풀마다 다르므로 함께 보낸다.
                "currency": pool_currency,
            }
        )
    return charts
