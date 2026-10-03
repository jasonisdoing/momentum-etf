"""종목풀 위험·수익 산점도 — 종목별 CAGR(연 수익률)과 MDD(최대 낙폭).

`/pools-risk-return` 화면의 단일 소스다. 종가 시계열은 순위·모멘텀 화면과 같은 공통 가격 경로
(가격 캐시 + `effective_prices` 가 붙이는 실시간 마지막 봉)이고, MDD 는 순위 화면이 쓰는 같은
함수(`perf_metrics.curve_metrics`)라 같은 기간이면 같은 값이다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.strategy.scoring import is_new_listing
from utils.perf_metrics import annualized_return_pct, calmar_ratio, curve_metrics


def is_short_listed(close: pd.Series, months: int) -> bool:
    """선택한 기간(N개월)보다 상장이 짧은 종목인가 — 기간이 같지 않은 종목을 한 차트에 섞지 않으려고 뺀다.

    순위 화면의 🆕 판정과 같은 함수다.
    """
    return is_new_listing(close, window_months=months)


def risk_return_point(close: pd.Series, months: int) -> dict[str, Any] | None:
    """한 종목의 CAGR·MDD. 구간에 봉이 2개 미만이거나 연환산할 수 없으면 None(점을 찍지 않는다).

    CAGR 은 달력 일수로 연환산한다 — 시작 종가에서 마지막 종가까지의 배수를 `365/일수` 제곱.
    """
    series = pd.to_numeric(close, errors="coerce").dropna()
    series = series[series > 0]
    if len(series) < 2:
        return None
    target = series.loc[series.index[-1] - pd.DateOffset(months=months) :]
    if len(target) < 2:
        return None

    start_value = float(target.iloc[0])
    values = target.iloc[1:].to_numpy()
    cagr_pct = annualized_return_pct(start_value, float(values[-1]), int((target.index[-1] - target.index[0]).days))
    if cagr_pct is None:
        return None
    mdd_pct = float(curve_metrics(start_value, values)["mdd_pct"])
    calmar = calmar_ratio(cagr_pct, mdd_pct)
    return {
        "cagr_pct": round(cagr_pct, 2),
        "mdd_pct": round(mdd_pct, 2),
        "calmar": None if calmar is None else round(calmar, 2),
    }


def compute_pool_risk_return(pool_id: str, months: int) -> dict[str, Any]:
    """풀의 종목을 점으로. 기간을 골랐으면 그 기간보다 상장이 짧은 종목은 뺀다.

    `excluded` 는 데이터가 모자라 점을 못 찍은 종목 수이고, 상장이 짧아 뺀 종목은 세지 않는다.
    """
    from utils.cache_utils import load_cached_frames_bulk_from_ticker_types
    from utils.effective_prices import apply_realtime_closes
    from utils.portfolio_io import load_all_holding_tickers
    from utils.settings_loader import get_ticker_type_settings
    from utils.slot_positions import _live_quotes
    from utils.stock_list_io import get_etfs

    settings = get_ticker_type_settings(pool_id) or {}
    country = str(settings.get("country_code") or "").strip().lower()
    items = get_etfs(pool_id)
    name_by = {str(item["ticker"]).strip(): str(item.get("name") or "").strip() for item in items if item.get("ticker")}
    tickers = list(name_by)

    frames = load_cached_frames_bulk_from_ticker_types([pool_id], tickers)
    close_by = {
        ticker: pd.to_numeric(frame["Close"], errors="coerce")
        for ticker, frame in frames.items()
        if frame is not None and not frame.empty and "Close" in frame
    }
    close_frame = pd.DataFrame(close_by).sort_index()

    # 순위·모멘텀과 같은 공통 경로로 실시간 마지막 봉을 붙인다.
    quotes = _live_quotes(pool_id, tickers)
    if quotes["by_ticker"] and not close_frame.empty:
        live = {ticker: quote["price"] for ticker, quote in quotes["by_ticker"].items()}
        close_frame = apply_realtime_closes(close_frame, live, quotes["session_ts"])

    held = load_all_holding_tickers(country_code=country or None)
    points: list[dict[str, Any]] = []
    short_listed = 0
    for ticker in tickers:
        if ticker in close_frame.columns and is_short_listed(close_frame[ticker], months):
            short_listed += 1
            continue
        metrics = risk_return_point(close_frame[ticker], months) if ticker in close_frame.columns else None
        if metrics is None:
            continue
        points.append(
            {
                "ticker": _display_ticker(ticker, country),
                "name": name_by[ticker] or ticker,
                **metrics,
                "is_held": ticker.strip().upper() in held,
            }
        )
    return {
        "pool_id": pool_id,
        "months": months,
        "points": points,
        "excluded": len(tickers) - short_listed - len(points),
    }


def _display_ticker(ticker: str, country: str) -> str:
    """상세 모달이 받는 표준 표기 — 호주는 `ASX:` 접두사를 붙인다(미국에 같은 티커가 있다)."""
    if country == "au":
        from utils.asx_ticker import ensure_asx_prefix

        return ensure_asx_prefix(ticker)
    return ticker
