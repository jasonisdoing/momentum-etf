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

from config import (
    MARKET_TREND_REGIME_SLOPE_DEADBAND,
    MARKET_TREND_REGIME_SLOPE_WINDOW,
    MARKET_TREND_SCORE_ANCHOR_PERCENTILE,
    TRADING_DAYS_PER_MONTH,
)
from utils.moving_averages import calculate_moving_average

logger = logging.getLogger(__name__)

# 표시 순서대로 정의. yf_ticker 는 Yahoo Finance 인덱스 심볼.
# kor_naver_symbol 이 있으면 한국 인덱스로 간주하고 가격은 네이버에서 받는다.
INDICES: list[dict[str, str]] = [
    {"name": "코스피", "yf_ticker": "^KS11", "kor_naver_symbol": "KOSPI"},
    {"name": "코스피 200", "yf_ticker": "^KS200", "kor_naver_symbol": "KPI200"},
    {"name": "S&P 500", "yf_ticker": "^GSPC"},
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


def _fetch_yf_intraday_last_close(yf_ticker: str) -> tuple[pd.Timestamp, float] | None:
    """yfinance intraday 1m 으로 오늘 ET 거래일의 가장 최근 종가를 반환.

    Yahoo Finance daily 데이터가 정규장 마감 후에도 수시간~다음날까지 갱신되지 않는
    지연이 종종 발생한다 (관측: KST 2026-06-10 시점 ET 6월 9일 row 가 NaN). 그 경우
    intraday 1분봉은 정상 마감가가 들어와 있으므로, 그 값으로 daily 마지막을 보강한다.

    실패 시 None — 호출자는 보강 없이 기존 daily 시리즈를 그대로 사용한다.
    """
    try:
        df = yf.Ticker(yf_ticker).history(period="1d", interval="1m")
    except Exception as exc:
        logger.warning("intraday 보강 호출 실패 (%s): %s", yf_ticker, exc)
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if close.empty:
        return None
    return pd.Timestamp(close.index[-1]), float(close.iloc[-1])


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
            name, ticker, price, change_pct, trend_pct, trend_score,
            current_regime, current_regime_days, prev_regime_1..3,
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
        # 12개월 정규화 점수 (-100 ~ +100, 화면 표시용)
        "trend_score": None,
        # 점수 환산 기준 (참조용)
        "score_range_high": None,
        "score_range_low": None,
        # 현재 레짐 + 지속 일수 + 직전 3개 레짐 기간 (테이블 표시용)
        "current_regime": None,
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

        # 미국 인덱스의 daily 마지막 종가가 Yahoo daily 갱신 지연으로 누락된 경우,
        # intraday 1분봉 마감가로 보강한다. (한국 인덱스는 네이버라서 적용 안 함.)
        intraday = _fetch_yf_intraday_last_close(yf_ticker)
        if intraday is not None and close_series is not None and not close_series.empty:
            try:
                intraday_ts, intraday_close = intraday
                intraday_date = (
                    intraday_ts.tz_convert(None).normalize()
                    if intraday_ts.tz is not None
                    else intraday_ts.normalize()
                )
                last_ts = pd.Timestamp(close_series.index[-1])
                last_date = (
                    last_ts.tz_convert(None).normalize()
                    if last_ts.tz is not None
                    else last_ts.normalize()
                )
                if intraday_date > last_date:
                    close_series = pd.concat(
                        [close_series, pd.Series([intraday_close], index=[intraday_date])]
                    )
            except Exception as exc:
                logger.warning("intraday 보강 머지 실패 (%s): %s", yf_ticker, exc)

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

    # 최근 12개월 일별 레짐을 계산해 연속 구간으로 그룹화 → 현재 지속일수 + 직전 3개 레짐 기간.
    ranges = _build_daily_regime_ranges(close_series, ma_series)
    if ranges:
        base["current_regime"] = ranges[-1]["regime"]
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

    # MA 괴리율 0%를 0점으로 두고, 12개월 상위 5%(95퍼센타일)/하위 5%(5퍼센타일) 괴리율로
    # 점수 정규화한다. 단발 극단치(최대/최소)는 천장을 한 순간만 만들어 +100 이 거의 안 찍히므로,
    # 상위 5% 구간에 들면 +100 에 도달하도록 퍼센타일을 앵커로 쓴다.
    score_window = TRADING_DAYS_PER_MONTH * 12
    trend_series_12m = _trend_pct_series(close_series, ma_series, score_window)
    valid_12m = [v for v in trend_series_12m if v is not None]
    if valid_12m:
        series_12m = pd.Series(valid_12m, dtype="float64")
        upper_q = MARKET_TREND_SCORE_ANCHOR_PERCENTILE / 100.0
        score_min = float(series_12m.quantile(1.0 - upper_q))
        score_max = float(series_12m.quantile(upper_q))
        base["score_range_high"] = score_max
        base["score_range_low"] = score_min
        base["trend_score"] = _normalize_score(base["trend_pct"], score_min, score_max)

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
    ranges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    strengthening_prev: bool | None = None
    for idx in range(start_idx, length):
        date_value = close_series.index[idx]
        date_str = (
            date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        )
        trend = full_trend[idx]
        slope = _trend_slope(full_trend, idx, MARKET_TREND_REGIME_SLOPE_WINDOW)
        regime, strengthening_prev = _regime_from_slope(
            trend, slope, strengthening_prev, MARKET_TREND_REGIME_SLOPE_DEADBAND
        )
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


def _trend_slope(full_trend: list[float | None], end_idx: int, window: int) -> float | None:
    """end_idx 까지 최근 ``window`` 거래일 추세%에 최소제곱 직선을 적합한 기울기(%/일).

    유효 점이 2개 미만이면 None. x 는 거래일 인덱스(결측은 건너뛰되 간격 보존).
    """
    lo = max(0, end_idx - int(window) + 1)
    pts = [(k, full_trend[k]) for k in range(lo, end_idx + 1) if full_trend[k] is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    mean_x = sum(x for x, _ in pts) / n
    mean_y = sum(y for _, y in pts) / n
    den = sum((x - mean_x) ** 2 for x, _ in pts)
    if den == 0:
        return None
    num = sum((x - mean_x) * (y - mean_y) for x, y in pts)
    return num / den


def _regime_from_slope(
    trend: float | None,
    slope: float | None,
    prev_strengthening: bool | None,
    deadband: float,
) -> tuple[str | None, bool | None]:
    """추세% 부호(방향) × 회귀 기울기(가속/감속, 데드밴드 히스테리시스)로 4단계 레짐 분류.

    기울기 > +deadband → 강화, < −deadband → 약화, 그 사이면 직전 강화/약화 상태 유지.
        MA 위(추세≥0) + 강화 → accel_up   (상승)
        MA 위        + 약화 → decel_up   (조정)
        MA 아래      + 강화 → decel_down (진정)
        MA 아래      + 약화 → accel_down (하락)
    반환: (regime, 갱신된 strengthening 상태)
    """
    if trend is None:
        return None, prev_strengthening
    if slope is None:
        strengthening = prev_strengthening if prev_strengthening is not None else (trend >= 0)
    elif slope > deadband:
        strengthening = True
    elif slope < -deadband:
        strengthening = False
    else:
        strengthening = prev_strengthening if prev_strengthening is not None else (slope >= 0)
    if trend >= 0:
        regime = "accel_up" if strengthening else "decel_up"
    else:
        regime = "decel_down" if strengthening else "accel_down"
    return regime, strengthening


def compute_index_history(yf_ticker: str, ma_type: str, ma_months: int) -> dict[str, Any]:
    """단일 지수의 최근 12개월 가격/추세 히스토리 + 각 일자별 레짐을 반환한다 (행 펼침용).

    Returns:
        ``{"ticker", "name", "ma_type", "ma_months",
            "history": [{date, close, ma, trend_pct, trend_score, regime}, ...],
            "trend_min_12m", "trend_max_12m"}``
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
        "trend_min_12m": None,
        "trend_max_12m": None,
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
        return empty_payload

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
    # 추세점수 정규화 앵커는 표(_build_item)와 동일하게 — 최신 시점 기준 트레일링 12개월
    # 퍼센타일(config) 을 쓴다. (이전엔 5년 min/max 라 표와 점수가 어긋났다.)
    score_window = TRADING_DAYS_PER_MONTH * 12
    anchor_window = [v for v in full_trend[max(0, length - score_window):length] if v is not None]
    if anchor_window:
        anchor_series = pd.Series(anchor_window, dtype="float64")
        upper_q = MARKET_TREND_SCORE_ANCHOR_PERCENTILE / 100.0
        score_min = float(anchor_series.quantile(1.0 - upper_q))
        score_max = float(anchor_series.quantile(upper_q))
    else:
        score_min = None
        score_max = None

    history: list[dict[str, Any]] = []
    strengthening_prev: bool | None = None
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

        # 레짐: 추세% 회귀 기울기 + 데드밴드(히스테리시스)로 분류.
        slope = _trend_slope(full_trend, idx, MARKET_TREND_REGIME_SLOPE_WINDOW)
        regime, strengthening_prev = _regime_from_slope(
            trend, slope, strengthening_prev, MARKET_TREND_REGIME_SLOPE_DEADBAND
        )

        history.append(
            {
                "date": date_str,
                "close": close,
                "ma": ma_v,
                "trend_pct": trend,
                "trend_score": trend_score,
                "regime": regime,
            }
        )

    return {
        "ticker": yf_ticker,
        "name": name,
        "ma_type": ma_type,
        "ma_months": int(ma_months),
        "history": history,
        "trend_min_12m": score_min,
        "trend_max_12m": score_max,
    }


__all__ = ["compute_market_trend", "compute_index_history", "INDICES"]
