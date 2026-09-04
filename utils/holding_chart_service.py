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

from config import HOLDING_CHART_MONTHS, HOLDING_CHART_SHOW_AVG_BUY_PRICE
from utils.price_series import positive_prices as _positive

_CANDLE_KEYS = ("Open", "High", "Low", "Close")


def holding_charts(
    pool: str,
    tickers: list[str],
    ma_days_list: list[int],
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
    # 내 평균 매입가 — 전 계좌 합산. 같은 티커가 다른 시장에도 있으면(IOO) 이 풀의 통화만 센다.
    # 끄면(config.HOLDING_CHART_SHOW_AVG_BUY_PRICE) 계좌를 아예 읽지 않는다.
    pool_currency = str((get_ticker_type_settings(pool) or {}).get("currency") or "").strip()
    avg_buy_by = (
        average_buy_price_by_ticker(wanted, currency=pool_currency or None) if HOLDING_CHART_SHOW_AVG_BUY_PRICE else {}
    )

    # 한 요청의 차트들이 **같은 날짜 축**을 쓰도록, 이 풀의 거래일을 먼저 모은다.
    # 이게 없으면 상장한 지 얼마 안 된 종목이 캔들 열몇 개로 가로 폭을 다 채워, 다른 종목과
    # 같은 기간을 보는 것처럼 보인다(실제로는 2주치인데 6개월치처럼 보인다).
    all_dates: set[pd.Timestamp] = set()
    for frame in frames.values():
        if frame is None or frame.empty:
            continue
        all_dates.update(frame.index)
    window_dates: list[pd.Timestamp] = []
    if all_dates:
        ordered_dates = sorted(all_dates)
        window_start = ordered_dates[-1] - pd.DateOffset(months=months)
        window_dates = [day for day in ordered_dates if day >= window_start]
    date_position = {day: position for position, day in enumerate(window_dates)}

    charts: list[dict[str, Any]] = []
    for ticker in wanted:
        frame = frames.get(ticker)
        if frame is None or frame.empty or any(key not in frame for key in _CANDLE_KEYS):
            continue
        cols = {key: _positive(frame[key]) for key in _CANDLE_KEYS}
        close = cols["Close"]
        # 화면이 보는 구간만 잘라 보내되, 이평선은 잘린 앞부분까지 써서 계산한다.
        ma_by_days = {days: close.rolling(days, min_periods=days).mean() for days in ma_days_list}
        span = frame.index[frame.index >= frame.index[-1] - pd.DateOffset(months=months)]
        if window_dates:
            # 공용 창 밖(이 종목만 더 과거를 들고 있는 경우)은 잘라 축을 맞춘다.
            span = span[span >= window_dates[0]]

        candles: list[dict[str, Any]] = []
        candle_dates: list[pd.Timestamp] = []
        points_by_days: dict[int, list[dict[str, Any]]] = {days: [] for days in ma_days_list}
        for day in span:
            values = [cols[key].get(day) for key in _CANDLE_KEYS]
            if any(pd.isna(value) for value in values):
                continue
            date = str(day.date())
            candles.append(dict(zip(("open", "high", "low", "close"), (float(v) for v in values)), time=date))
            candle_dates.append(day)
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
                # 공용 날짜 축 — 화면이 보이는 구간을 이 값으로 잡는다.
                #   window_bars  이 창의 전체 거래일 수
                #   leading_bars 창 시작부터 이 종목의 첫 캔들까지 비어 있는 칸 수
                # 신규 상장 종목은 leading_bars 가 커서, 캔들이 오른쪽 일부만 채우고 왼쪽은 빈다.
                "window_bars": len(window_dates),
                "leading_bars": date_position.get(candle_dates[0], 0) if window_dates and candle_dates else 0,
            }
        )
    return charts
