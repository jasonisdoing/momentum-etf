"""시장지수 추세 데이터 서비스.

코스피/코스피200/S&P500/나스닥/나스닥100 의 현재가, 변동률, MA 대비 추세% 를 계산한다.
가격 소스:
    - 한국 인덱스(KOSPI/KOSPI200): 네이버 차트 API (yfinance 가 1거래일 지연되는 이슈 회피)
    - 미국 인덱스(S&P500/나스닥/나스닥100): yfinance
MA 계산은 utils.moving_averages 사용.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from config import TRADING_DAYS_PER_MONTH
from utils.moving_averages import calculate_moving_average

logger = logging.getLogger(__name__)

# 표시 순서대로 정의. yf_ticker 는 Yahoo Finance 인덱스 심볼.
# kor_naver_symbol 이 있으면 한국 인덱스로 간주하고 가격은 네이버에서 받는다.
INDICES: list[dict[str, str]] = [
    {"name": "코스피", "yf_ticker": "^KS11", "kor_naver_symbol": "KOSPI"},
    {"name": "코스피 200", "yf_ticker": "^KS200", "kor_naver_symbol": "KPI200"},
    {"name": "S&P 500", "yf_ticker": "^GSPC"},
    {"name": "나스닥", "yf_ticker": "^IXIC"},
    {"name": "나스닥 100", "yf_ticker": "^NDX"},
]

# 네이버 차트 (legacy XML) — 인덱스 일봉 OHLCV. count 만큼 최신부터 거꾸로 N건 반환.
_NAVER_INDEX_CHART_URL = "https://fchart.stock.naver.com/sise.nhn"
_NAVER_ITEM_RE = re.compile(r'<item data="([^"]+)"')


def _fetch_naver_kor_index_close(symbol: str, count: int) -> pd.Series | None:
    """네이버 차트 API 에서 한국 인덱스 일봉 종가 시계열을 받아온다.

    Args:
        symbol: KOSPI 또는 KPI200.
        count: 최근부터 N 거래일.

    Returns:
        DatetimeIndex 정렬된 pd.Series(Close) 또는 None (실패 시).
    """
    try:
        resp = requests.get(
            _NAVER_INDEX_CHART_URL,
            params={"symbol": symbol, "timeframe": "day", "count": int(count), "requestType": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.encoding = "EUC-KR"
        items = _NAVER_ITEM_RE.findall(resp.text)
    except Exception:
        logger.exception("네이버 인덱스 차트 조회 실패: %s", symbol)
        return None
    if not items:
        return None

    dates: list[pd.Timestamp] = []
    closes: list[float] = []
    for raw in items:
        parts = raw.split("|")
        if len(parts) < 5:
            continue
        try:
            ts = pd.Timestamp(parts[0])
            close = float(parts[4])
        except (ValueError, TypeError):
            continue
        dates.append(ts)
        closes.append(close)
    if not dates:
        return None
    series = pd.Series(closes, index=pd.DatetimeIndex(dates))
    return series.sort_index()

# 과거 시점 추세 계산용 trading-day offset (오늘 대비 N 거래일 전).
TRADING_DAYS_PER_WEEK = 5
TREND_OFFSETS_DAYS = {
    "trend_pct_w1": TRADING_DAYS_PER_WEEK * 1,            # 1주 전
    "trend_pct_w2": TRADING_DAYS_PER_WEEK * 2,            # 2주 전
    "trend_pct_w3": TRADING_DAYS_PER_WEEK * 3,            # 3주 전
    "trend_pct_w4": TRADING_DAYS_PER_WEEK * 4,            # 4주 전
}


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def compute_market_trend(ma_type: str, ma_months: int) -> dict[str, Any]:
    """5개 시장지수의 현재가/변동률/MA 추세%(현재 + 과거 3시점) 를 계산해 반환한다.

    Args:
        ma_type: SMA/EMA/WMA/DEMA/TEMA/HMA/ALMA
        ma_months: 1~12 (정수). 내부에서 ``ma_months * TRADING_DAYS_PER_MONTH`` 일로 환산.

    Returns:
        ``{"ma_type", "ma_months", "items": [{
            name, ticker, price, change_pct,
            trend_pct, trend_pct_w1, trend_pct_m1, trend_pct_m3,
        }, ...]}``
    """

    ma_days = int(ma_months) * int(TRADING_DAYS_PER_MONTH)
    if ma_days < 2:
        ma_days = 2

    # 미국 인덱스만 yfinance 로 일괄 다운로드 (한국 2개는 네이버 사용).
    us_tickers = [idx["yf_ticker"] for idx in INDICES if not idx.get("kor_naver_symbol")]
    try:
        df = yf.download(
            tickers=us_tickers,
            period="2y",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        logger.exception("yfinance 시장지수 다운로드 실패")
        df = None

    # 한국 인덱스(KOSPI/KPI200) 종가는 네이버 차트 API 로 조회 (2년치 ≈ 500거래일).
    kor_close_by_ticker: dict[str, pd.Series] = {}
    for idx in INDICES:
        naver_symbol = idx.get("kor_naver_symbol")
        if not naver_symbol:
            continue
        series = _fetch_naver_kor_index_close(naver_symbol, count=500)
        if series is not None and not series.empty:
            kor_close_by_ticker[idx["yf_ticker"]] = series

    items: list[dict[str, Any]] = []
    for idx in INDICES:
        kor_close = kor_close_by_ticker.get(idx["yf_ticker"])
        item = _build_item(df, idx["yf_ticker"], idx["name"], ma_days, ma_type, kor_close)
        items.append(item)

    return {
        "ma_type": ma_type,
        "ma_months": int(ma_months),
        "items": items,
    }


def _build_item(
    df: pd.DataFrame | None,
    yf_ticker: str,
    name: str,
    ma_days: int,
    ma_type: str,
    kor_close: pd.Series | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "ticker": yf_ticker,
        "price": None,
        "change_pct": None,
        # 원본 추세 % (MA 괴리율)
        "trend_pct": None,
        "trend_pct_w1": None,
        "trend_pct_w2": None,
        "trend_pct_w3": None,
        "trend_pct_w4": None,
        # 12개월 정규화 점수 (-100 ~ +100, 화면 표시용)
        "trend_score": None,
        "trend_score_w1": None,
        "trend_score_w2": None,
        "trend_score_w3": None,
        "trend_score_w4": None,
        # 점수 환산 기준 (참조용)
        "score_range_high": None,
        "score_range_low": None,
        # 현재 레짐의 지속 일수 + 직전 3개 레짐 기간 (테이블 표시용)
        "current_regime_days": None,
        "prev_regime_1": None,
        "prev_regime_2": None,
        "prev_regime_3": None,
    }
    # 한국 인덱스는 네이버에서 받은 close_series 를 우선 사용한다.
    if kor_close is not None and not kor_close.empty:
        close_series = kor_close.dropna()
    elif df is None or df.empty:
        return base
    else:
        # multi-ticker 결과는 컬럼 멀티인덱스(ticker, ohlc). 단일 ticker 결과는 평탄.
        try:
            if (yf_ticker, "Close") in df.columns:
                close_series = df[(yf_ticker, "Close")].dropna()
            elif "Close" in df.columns:
                close_series = df["Close"].dropna()
            else:
                return base
        except Exception:
            return base

    if close_series is None or close_series.empty or len(close_series) < 2:
        return base

    latest_price = _to_float(close_series.iloc[-1])
    prev_price = _to_float(close_series.iloc[-2])
    if latest_price is None or prev_price is None:
        return base

    change_pct: float | None = None
    if prev_price != 0:
        change_pct = (latest_price / prev_price - 1.0) * 100.0

    base["price"] = latest_price
    base["change_pct"] = change_pct

    # MA 시리즈는 전체 가격 시리즈에 대해 한 번만 계산하고, 시점별로 인덱싱한다.
    try:
        ma_series = calculate_moving_average(close_series, ma_days, ma_type)
    except Exception:
        logger.exception("MA 계산 실패: %s (type=%s, days=%d)", yf_ticker, ma_type, ma_days)
        return base

    base["trend_pct"] = _trend_pct_at(close_series, ma_series, offset=0)
    for key, offset in TREND_OFFSETS_DAYS.items():
        base[key] = _trend_pct_at(close_series, ma_series, offset=offset)

    # 최근 12개월 일별 레짐을 계산해 연속 구간으로 그룹화 → 현재 지속일수 + 직전 3개 레짐 기간.
    ranges = _build_daily_regime_ranges(close_series, ma_series)
    if ranges:
        base["current_regime_days"] = ranges[-1]["days"]
        for slot, offset in ((1, -2), (2, -3), (3, -4)):
            if len(ranges) >= -offset:
                r = ranges[offset]
                base[f"prev_regime_{slot}"] = {
                    "regime": r["regime"],
                    "start_date": r["start_date"],
                    "end_date": r["end_date"],
                    "days": r["days"],
                }

    # MA 괴리율 0%를 0점으로 두고, 12개월 위쪽 최고/아래쪽 최저 괴리율로 점수 정규화.
    score_window = TRADING_DAYS_PER_MONTH * 12
    trend_series_12m = _trend_pct_series(close_series, ma_series, score_window)
    valid_12m = [v for v in trend_series_12m if v is not None]
    if valid_12m:
        score_min = min(valid_12m)
        score_max = max(valid_12m)
        base["score_range_high"] = score_max
        base["score_range_low"] = score_min
        base["trend_score"] = _normalize_score(base["trend_pct"], score_min, score_max)
        base["trend_score_w1"] = _normalize_score(base["trend_pct_w1"], score_min, score_max)
        base["trend_score_w2"] = _normalize_score(base["trend_pct_w2"], score_min, score_max)
        base["trend_score_w3"] = _normalize_score(base["trend_pct_w3"], score_min, score_max)
        base["trend_score_w4"] = _normalize_score(base["trend_pct_w4"], score_min, score_max)

    return base


def _trend_pct_series(
    close_series: pd.Series,
    ma_series: pd.Series,
    take_days: int,
) -> list[float | None]:
    """최근 ``take_days`` 거래일의 추세% 시리즈 (오래된 → 최신)."""
    if close_series is None or ma_series is None:
        return []
    length = min(len(close_series), len(ma_series))
    take = min(length, int(take_days))
    out: list[float | None] = []
    for i in range(-take, 0):
        price = _to_float(close_series.iloc[i])
        ma_v = _to_float(ma_series.iloc[i])
        if price is None or ma_v is None or ma_v == 0:
            out.append(None)
        else:
            out.append((price / ma_v - 1.0) * 100.0)
    return out


def _normalize_score(value: float | None, lo: float, hi: float) -> float | None:
    """MA 괴리율 0%를 0점으로 고정하고 위/아래 영역을 따로 정규화한다."""
    if value is None:
        return None
    if value == 0:
        return 0.0
    if value > 0:
        if hi <= 0:
            return 0.0
        return max(0.0, min(100.0, value / hi * 100.0))
    if lo >= 0:
        return 0.0
    return max(-100.0, min(0.0, value / abs(lo) * 100.0))


def _trend_pct_at(
    close_series: pd.Series,
    ma_series: pd.Series,
    *,
    offset: int,
) -> float | None:
    """``offset`` 거래일 전의 (종가 / MA - 1) * 100 을 반환한다. offset=0 이면 최신."""
    if close_series is None or ma_series is None:
        return None
    length = min(len(close_series), len(ma_series))
    if length <= offset:
        return None
    idx = -1 - offset
    price = _to_float(close_series.iloc[idx])
    ma_value = _to_float(ma_series.iloc[idx])
    if price is None or ma_value is None or ma_value == 0:
        return None
    return (price / ma_value - 1.0) * 100.0


def _build_daily_regime_ranges(
    close_series: pd.Series,
    ma_series: pd.Series,
    window_days: int = TRADING_DAYS_PER_MONTH * 12,
) -> list[dict[str, Any]]:
    """최근 ``window_days`` 거래일의 일별 레짐을 계산해 연속 구간으로 그룹화한다.

    반환 형식: [{"regime": str, "start_date": str, "end_date": str, "days": int}, ...]
    오래된 순서 → 최신 순서.
    """
    if close_series is None or ma_series is None:
        return []
    length = min(len(close_series), len(ma_series))
    if length == 0:
        return []

    # 일별 추세% (전체 시리즈)
    full_trend: list[float | None] = []
    for idx in range(length):
        c = _to_float(close_series.iloc[idx])
        m = _to_float(ma_series.iloc[idx])
        if c is None or m is None or m == 0:
            full_trend.append(None)
        else:
            full_trend.append((c / m - 1.0) * 100.0)

    take = min(length, int(window_days))
    start_idx = length - take
    week_offsets = (5, 10, 15, 20)
    ranges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for idx in range(start_idx, length):
        date_value = close_series.index[idx]
        date_str = (
            date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        )
        trend = full_trend[idx]
        past_vals = []
        for off in week_offsets:
            j = idx - off
            if j >= 0 and full_trend[j] is not None:
                past_vals.append(full_trend[j])
        avg_past = sum(past_vals) / 4 if len(past_vals) == 4 else None
        regime = _classify_regime(trend, avg_past)
        if regime is None:
            if current:
                ranges.append(current)
                current = None
            continue
        if current is None or current["regime"] != regime:
            if current:
                ranges.append(current)
            current = {
                "regime": regime,
                "start_date": date_str,
                "end_date": date_str,
                "days": 1,
            }
        else:
            current["end_date"] = date_str
            current["days"] += 1
    if current:
        ranges.append(current)
    return ranges


def _classify_regime(trend: float | None, avg_past: float | None) -> str | None:
    """현재 추세%와 1·2·3·4주 전 추세% 평균으로 4단계 레짐을 분류한다.

    규칙:
        delta = trend - avg_past
        trend >= 0  AND  delta >= 0  →  accel_up   (상승: MA 위 + 평균보다 강함)
        trend >= 0  AND  delta < 0   →  decel_up   (조정: MA 위 + 평균보다 약함)
        trend < 0   AND  delta > 0   →  decel_down (진정: MA 아래 + 평균보다 강함)
        trend < 0   AND  delta <= 0  →  accel_down (하락: MA 아래 + 평균보다 약함/유지)
    """
    if trend is None or avg_past is None:
        return None
    delta = trend - avg_past
    if trend >= 0:
        return "accel_up" if delta >= 0 else "decel_up"
    return "decel_down" if delta > 0 else "accel_down"


def compute_index_history(yf_ticker: str, ma_type: str, ma_months: int) -> dict[str, Any]:
    """단일 지수의 최근 12개월 가격/추세 히스토리 + 각 일자별 레짐을 반환한다 (행 펼침용).

    Returns:
        ``{"ticker", "name", "ma_type", "ma_months",
            "history": [{date, close, ma, trend_pct, regime}, ...],
            "week_markers": [{week: 1, date, trend_pct}, ...]}``
        해당 ticker 가 알려진 인덱스가 아니면 name 은 ticker 그대로 사용.
    """
    index_meta = next((idx for idx in INDICES if idx["yf_ticker"] == yf_ticker), None)
    name = index_meta["name"] if index_meta else yf_ticker
    naver_symbol = (index_meta or {}).get("kor_naver_symbol")

    ma_days = int(ma_months) * int(TRADING_DAYS_PER_MONTH)
    if ma_days < 2:
        ma_days = 2

    empty_payload = {
        "ticker": yf_ticker,
        "name": name,
        "ma_type": ma_type,
        "ma_months": int(ma_months),
        "history": [],
        "week_markers": [],
    }

    close_series: pd.Series | None = None
    if naver_symbol:
        # 한국 인덱스: 네이버 차트에서 직접 받는다 (5년 ≈ 1250거래일, 여유 포함 1500).
        close_series = _fetch_naver_kor_index_close(naver_symbol, count=1500)
        if close_series is None:
            return empty_payload
    else:
        try:
            df = yf.download(
                tickers=yf_ticker,
                period="10y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception:
            logger.exception("yfinance 단일 인덱스 다운로드 실패: %s", yf_ticker)
            df = None

        if df is None or df.empty:
            return empty_payload

        # yfinance 가 단일 ticker 라도 컬럼을 멀티인덱스로 줄 수 있어 평탄화.
        close_raw = df["Close"] if "Close" in df.columns else None
        if close_raw is None:
            try:
                close_raw = df.xs("Close", axis=1, level=0)
            except Exception:
                close_raw = None
        if close_raw is None:
            return empty_payload
        if isinstance(close_raw, pd.DataFrame):
            close_raw = close_raw.iloc[:, 0]
        close_series = close_raw.dropna()
    if len(close_series) < 2:
        return {
            "ticker": yf_ticker,
            "name": name,
            "ma_type": ma_type,
            "ma_months": int(ma_months),
            "history": [],
            "week_markers": [],
        }

    try:
        ma_series = calculate_moving_average(close_series, ma_days, ma_type)
    except Exception:
        logger.exception("MA 계산 실패: %s (type=%s, days=%d)", yf_ticker, ma_type, ma_days)
        ma_series = None

    # 최근 5년치 = 약 1200 거래일 (프론트에서 1개월~5년 범위 선택 가능)
    tail = TRADING_DAYS_PER_MONTH * 12 * 5
    length = min(len(close_series), len(ma_series) if ma_series is not None else len(close_series))
    take = min(length, tail)

    # 전체 시리즈에 대한 일별 trend% 사전 계산 (인덱스 = 0..length-1).
    full_trend: list[float | None] = []
    for idx in range(length):
        c = _to_float(close_series.iloc[idx])
        m = _to_float(ma_series.iloc[idx]) if ma_series is not None else None
        if c is None or m is None or m == 0:
            full_trend.append(None)
        else:
            full_trend.append((c / m - 1.0) * 100.0)

    start = length - take
    valid_trend_window = [v for v in full_trend[start:length] if v is not None]
    score_min = min(valid_trend_window) if valid_trend_window else None
    score_max = max(valid_trend_window) if valid_trend_window else None

    week_offsets = (5, 10, 15, 20)
    history: list[dict[str, Any]] = []
    for idx in range(start, length):
        date_value = close_series.index[idx]
        date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        close = _to_float(close_series.iloc[idx])
        ma_v = _to_float(ma_series.iloc[idx]) if ma_series is not None else None
        trend = full_trend[idx]
        trend_score = (
            _normalize_score(trend, score_min, score_max)
            if score_min is not None and score_max is not None
            else None
        )

        # 해당 일자의 1·2·3·4주 전 추세% 평균과 비교 → 4단계 레짐 분류 + delta.
        past_vals = []
        for off in week_offsets:
            j = idx - off
            if j >= 0 and full_trend[j] is not None:
                past_vals.append(full_trend[j])
        avg_past = sum(past_vals) / len(past_vals) if len(past_vals) == len(week_offsets) else None
        delta = (trend - avg_past) if (trend is not None and avg_past is not None) else None
        regime = _classify_regime(trend, avg_past)

        history.append(
            {
                "date": date_str,
                "close": close,
                "ma": ma_v,
                "trend_pct": trend,
                "trend_score": trend_score,
                "delta_pct": delta,
                "regime": regime,
            }
        )

    # 1~8주 마커 — history 시리즈에서 끝에서부터 5N 거래일 전. regime 도 포함.
    week_markers: list[dict[str, Any]] = []
    for week in range(1, 9):
        offset = TRADING_DAYS_PER_WEEK * week
        idx = len(history) - 1 - offset
        if idx < 0:
            continue
        point = history[idx]
        week_markers.append({
            "week": week,
            "date": point["date"],
            "trend_pct": point["trend_pct"],
            "regime": point.get("regime"),
        })

    # 게이지 정규화용: 12개월 |delta| 최대값.
    valid_deltas = [p["delta_pct"] for p in history if p.get("delta_pct") is not None]
    delta_abs_max = max(abs(d) for d in valid_deltas) if valid_deltas else None

    # 12개월 추세% 범위 + 최신 4주 평균 (게이지 표시용)
    latest = history[-1] if history else None
    latest_avg_past = None
    if latest and latest.get("trend_pct") is not None and latest.get("delta_pct") is not None:
        latest_avg_past = latest["trend_pct"] - latest["delta_pct"]

    return {
        "ticker": yf_ticker,
        "name": name,
        "ma_type": ma_type,
        "ma_months": int(ma_months),
        "history": history,
        "week_markers": week_markers,
        "delta_abs_max": delta_abs_max,
        "trend_min_12m": score_min,
        "trend_max_12m": score_max,
        "latest_avg_past": latest_avg_past,
    }


__all__ = ["compute_market_trend", "compute_index_history", "INDICES"]
