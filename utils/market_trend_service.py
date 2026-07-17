"""시장지수 추세 데이터 서비스.

코스피/코스피200/S&P500/나스닥/나스닥100 의 현재가, 변동률, MA 대비 추세% 를 계산한다.
가격 소스:
    - 한국 인덱스(KOSPI/KOSPI200): 네이버 차트 API (yfinance 가 1거래일 지연되는 이슈 회피)
    - 미국 인덱스(S&P500/나스닥/나스닥100): yfinance
    - 나스닥 100 선물: 이력은 yfinance(NQ=F), 최신 봉은 토스(RFU.NQc1, REAL_TIME)로 보강
MA 계산은 utils.moving_averages 사용.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    MARKET_TREND_REGIME_CONFIRM_DAYS,
    MARKET_TREND_REGIME_MA_LONG,
    MARKET_TREND_REGIME_MA_SHORT,
    MARKET_TREND_SCORE_ANCHOR_PERCENTILE,
    MARKET_TREND_SUPERTREND_MULTIPLIER,
    MARKET_TREND_SUPERTREND_PERIOD,
    TRADING_DAYS_PER_MONTH,
)

logger = logging.getLogger(__name__)

# 표시 순서대로 정의. yf_ticker 는 Yahoo Finance 인덱스 심볼.
# kor_naver_symbol 이 있으면 한국 인덱스로 간주하고 가격은 네이버에서 받는다.
INDICES: list[dict[str, str]] = [
    {"name": "코스피", "yf_ticker": "^KS11", "kor_naver_symbol": "KOSPI"},
    {"name": "코스피 200", "yf_ticker": "^KS200", "kor_naver_symbol": "KPI200"},
    {"name": "다우존스", "yf_ticker": "^DJI"},
    {"name": "S&P 500", "yf_ticker": "^GSPC"},
    {"name": "나스닥 100", "yf_ticker": "^NDX"},
    {"name": "필라델피아 반도체", "yf_ticker": "^SOX"},
]

# 네이버 차트 (legacy XML) — 일봉 OHLCV 조회는 공통 헬퍼(utils/naver_chart.py)를 쓴다.


def _fetch_naver_kor_index_ohlc(symbol: str, count: int) -> pd.DataFrame | None:
    """네이버 차트 API 에서 한국 인덱스 일봉 OHLC 시계열을 받아온다.

    Returns:
        DatetimeIndex 정렬된 pd.DataFrame(columns=['Open', 'High', 'Low', 'Close']) 또는 None.
    """
    from utils.naver_chart import fetch_naver_daily_ohlc

    return fetch_naver_daily_ohlc(symbol, count)


def _fetch_naver_kor_index_close(symbol: str, count: int) -> pd.Series | None:
    """네이버 차트 API 에서 한국 인덱스 일봉 종가 시계열을 받아온다.

    Args:
        symbol: KOSPI 또는 KPI200.
        count: 최근부터 N 거래일.

    Returns:
        DatetimeIndex 정렬된 pd.Series(Close) 또는 None (실패 시).
    """
    df = _fetch_naver_kor_index_ohlc(symbol, count)
    if df is None or df.empty:
        return None
    return df["Close"]


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


# 토스 실시간 일봉으로 최신 종가를 보강할 심볼 (yfinance 지연 회피 — REAL_TIME 피드)
_TOSS_DAILY_OVERLAY = {"NQ=F": "RFU.NQc1"}


def _apply_toss_latest_overlay(close_series: pd.Series, toss_code: str) -> pd.Series | None:
    """토스 최신 일봉(형성 중 포함)으로 마지막 종가를 갱신/추가한다. 실패 시 None."""
    try:
        from services.toss_market_service import fetch_toss_latest_daily_close

        date_str, latest_close = fetch_toss_latest_daily_close(toss_code)
        latest_date = pd.Timestamp(date_str)
        series = close_series.copy()
        idx = pd.to_datetime(series.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        series.index = idx.normalize()
        if series.index[-1] == latest_date:
            series.iloc[-1] = latest_close
        elif series.index[-1] < latest_date:
            series = pd.concat([series, pd.Series([latest_close], index=[latest_date])])
        return series
    except Exception as exc:
        logger.warning("토스 최신 일봉 보강 실패 (%s): %s", toss_code, exc)
        return None


def _apply_intraday_boost(close_series: pd.Series | None, yf_ticker: str) -> pd.Series | None:
    """미국 인덱스 daily 마지막 종가가 Yahoo 갱신 지연으로 누락된 경우 최신 종가를 보강한다.

    나스닥 100 선물 등 토스 REAL_TIME 심볼은 토스 최신 일봉으로 갱신/추가(오늘 형성 중 봉 포함)하고,
    그 외/토스 실패 시엔 기존 Yahoo intraday 1분봉 마감가를 덧붙인다. 표(_build_item)와
    차트(compute_index_history)가 동일한 최신 종가를 쓰도록 공통 사용한다.
    """
    if close_series is None or close_series.empty:
        return close_series
    toss_code = _TOSS_DAILY_OVERLAY.get(yf_ticker)
    if toss_code:
        boosted = _apply_toss_latest_overlay(close_series, toss_code)
        if boosted is not None:
            return boosted
    intraday = _fetch_yf_intraday_last_close(yf_ticker)
    if intraday is None:
        return close_series
    try:
        intraday_ts, intraday_close = intraday
        intraday_date = (
            intraday_ts.tz_convert(None).normalize() if intraday_ts.tz is not None else intraday_ts.normalize()
        )
        last_ts = pd.Timestamp(close_series.index[-1])
        last_date = last_ts.tz_convert(None).normalize() if last_ts.tz is not None else last_ts.normalize()
        if intraday_date > last_date:
            return pd.concat([close_series, pd.Series([intraday_close], index=[intraday_date])])
    except Exception as exc:
        logger.warning("intraday 보강 머지 실패 (%s): %s", yf_ticker, exc)
    return close_series


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def compute_market_trend() -> dict[str, Any]:
    """5개 시장지수의 현재가/변동률/MA 추세%(현재 + 과거 3시점) 를 계산해 반환한다.

    MA는 SMA MARKET_TREND_REGIME_MA_SHORT일 고정.

    Returns:
        ``{"ma_days", "items": [{
            name, ticker, price, change_pct, trend_pct, trend_score,
            pct_from_high, current_regime, current_regime_days,
        }, ...]}``
    """
    ma_days = MARKET_TREND_REGIME_MA_SHORT

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

    # 한국 인덱스(KOSPI/KPI200) OHLC 는 네이버 차트 API 로 조회 (2년치 ≈ 500거래일).
    kor_ohlc_by_ticker: dict[str, pd.DataFrame] = {}
    for idx in INDICES:
        naver_symbol = idx.get("kor_naver_symbol")
        if not naver_symbol:
            continue
        ohlc = _fetch_naver_kor_index_ohlc(naver_symbol, count=500)
        if ohlc is not None and not ohlc.empty:
            kor_ohlc_by_ticker[idx["yf_ticker"]] = ohlc

    items: list[dict[str, Any]] = []
    for idx in INDICES:
        kor_ohlc = kor_ohlc_by_ticker.get(idx["yf_ticker"])
        item = _build_item(df, idx["yf_ticker"], idx["name"], ma_days, kor_ohlc)
        items.append(item)

    return {
        "ma_days": ma_days,
        "items": items,
    }


def _build_item(
    df: pd.DataFrame | None,
    yf_ticker: str,
    name: str,
    ma_days: int,
    kor_ohlc: pd.DataFrame | None = None,
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
        # 52주 전고점 대비 등락률 (현재가 ÷ 52주 최고 − 1) × 100 — 0 이하(고점=0)
        "pct_from_high": None,
        # 현재 레짐 + 지속 일수 (테이블 표시용)
        "current_regime": None,
        "current_regime_days": None,
        # 현재 레짐이 상승이 아닐 때: 마지막 상승 구간 종료 후 경과 거래일 (12개월 내 상승 없으면 None)
        "days_since_last_up": None,
        # 현재 레짐이 하락일 때: 마지막 중립 구간 종료 후 경과 거래일 (12개월 내 중립 없으면 None)
        "days_since_last_neutral": None,
    }
    # 한국 인덱스는 네이버에서 받은 OHLC 의 종가를 우선 사용한다.
    full_df = None
    kor_close = kor_ohlc["Close"] if (kor_ohlc is not None and "Close" in kor_ohlc.columns) else None
    if kor_close is not None and not kor_close.empty:
        full_df = kor_ohlc.copy()
        close_series = full_df["Close"].dropna()
    elif df is None or df.empty:
        return base
    else:
        # multi-ticker 결과는 컬럼 멀티인덱스(ticker, ohlc). 단일 ticker 결과는 평탄.
        try:
            if (yf_ticker, "Close") in df.columns:
                full_df = df.xs(yf_ticker, axis=1, level=0).copy()
                close_series = full_df["Close"].dropna()
            elif "Close" in df.columns:
                full_df = df.copy()
                close_series = full_df["Close"].dropna()
            else:
                return base
        except Exception:
            return base

        # 미국 인덱스 daily 마지막 종가가 지연 누락되면 intraday 마감가로 보강 (한국=네이버 제외).
        close_series = _apply_intraday_boost(close_series, yf_ticker)

        # 보정된 close_series 를 df 에 다시 반영 (인덱스가 확장되었을 경우 대비 reindex 적용)
        full_df = full_df.reindex(close_series.index)
        full_df["Close"] = close_series
        full_df = full_df.ffill()

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

    # 추세 점수는 새 레짐 정본과 맞춰 SMA MA20 대비 괴리율을 쓴다.
    ma_series = close_series.rolling(ma_days).mean()

    base["trend_pct"] = _trend_pct_at(close_series, ma_series, offset=0)

    # 52주(252거래일) 전고점 대비 등락률. 현재가가 고점이면 0, 아래면 음수.
    high_window = close_series.tail(TRADING_DAYS_PER_MONTH * 12 + 12)
    high_52w = _to_float(high_window.max()) if not high_window.empty else None
    if high_52w is not None and high_52w > 0 and latest_price is not None:
        base["pct_from_high"] = (latest_price / high_52w - 1.0) * 100.0

    # 최근 12개월 일별 레짐을 ST(SuperTrend)로 계산한다.
    try:
        st_period, st_multiplier = _resolve_supertrend_params(yf_ticker)
        st_df = _calculate_supertrend(full_df, period=st_period, multiplier=st_multiplier)
        st_dir = st_df["direction"]
    except Exception:
        st_dir = pd.Series(1, index=full_df.index)

    regime_series = pd.Series("accel_up", index=full_df.index, dtype=object)
    regime_series[st_dir == -1] = "accel_down"

    ranges = _build_regime_ranges_from_series(regime_series, TRADING_DAYS_PER_MONTH * 12)
    if ranges:
        base["current_regime"] = ranges[-1]["regime"]
        base["current_regime_days"] = ranges[-1]["days"]

        def _days_since_last(target_regime: str) -> int | None:
            elapsed = 0
            for seg in reversed(ranges):
                if seg["regime"] == target_regime:
                    return elapsed
                elapsed += int(seg["days"])
            return None

        if ranges[-1]["regime"] == "accel_down":
            base["days_since_last_up"] = _days_since_last("accel_up")
        else:
            base["days_since_last_neutral"] = _days_since_last("accel_down")

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


def _resolve_confirm_days(yf_ticker: str) -> int:
    """지수(yf_ticker)별 MA20/60 레짐 확인 거래일 수 N을 반환한다."""
    if yf_ticker not in MARKET_TREND_REGIME_CONFIRM_DAYS:
        raise ValueError(
            f"MARKET_TREND_REGIME_CONFIRM_DAYS 에 지수 '{yf_ticker}' 의 확인일 수가 등록되지 않았습니다. "
            "config.py 에 해당 지수를 등록해주세요."
        )
    value = int(MARKET_TREND_REGIME_CONFIRM_DAYS[yf_ticker])
    if value < 0:
        raise ValueError(f"MARKET_TREND_REGIME_CONFIRM_DAYS 값은 0 이상이어야 합니다: {yf_ticker}={value}")
    return value


def _resolve_supertrend_params(yf_ticker: str) -> tuple[int, float]:
    """차트 표시용 SuperTrend 기간/곱수를 반환한다."""
    if yf_ticker not in MARKET_TREND_SUPERTREND_MULTIPLIER:
        raise ValueError(
            f"MARKET_TREND_SUPERTREND_MULTIPLIER 에 지수 '{yf_ticker}' 의 곱수가 등록되지 않았습니다. "
            "config.py 에 해당 지수를 등록해주세요."
        )
    return MARKET_TREND_SUPERTREND_PERIOD, MARKET_TREND_SUPERTREND_MULTIPLIER[yf_ticker]


def _calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """차트 표시용 SuperTrend 지표를 계산한다."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]
        prev_close = close.iloc[i - 1]

        final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper else prev_upper
        final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower else prev_lower

        prev_dir = direction.iloc[i - 1]
        if prev_dir == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = final_upper.iloc[i]

    final_upper.iloc[0] = basic_upper.iloc[0]
    final_lower.iloc[0] = basic_lower.iloc[0]
    supertrend.iloc[0] = basic_lower.iloc[0]

    return pd.DataFrame({"supertrend": supertrend, "direction": direction}, index=df.index)


def is_market_trend_index(ticker: str) -> bool:
    """시장추세 지수 목록(INDICES)에 등록된 yf_ticker 인지 여부."""
    return any(idx["yf_ticker"] == ticker for idx in INDICES)


# MA20/60 레짐 백테스트 파라미터 (읽기 전용 분석 — DB 저장/운영 미반영)
REGIME_BACKTEST_MA_SHORT = 20  # MA 교차 방식 단기선
REGIME_BACKTEST_MA_LONG = 60   # MA 교차 방식 장기선
REGIME_BACKTEST_CONFIRM_MAX = 5  # MA 교차 확인 필터를 0~N일까지 나열
# 현금제어 기본 현금 비중(%) — 상승 0%, 중립 15%, 하락 30% (화면에서 조정)
REGIME_BACKTEST_DEFAULT_CASH = {"up": 0.0, "neutral": 15.0, "down": 30.0}


def _regime_series_ma_cross(close: pd.Series, ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
    """버퍼 없는 MA20/60 위치 기반 레짐(벡터화).

    - MA20 < MA60 → 하락
    - MA20 >= MA60 이면서 종가 > MA20 → 상승
    - MA20 >= MA60 이면서 종가 < MA60 → 하락
    - 그 외 → 중립
    """
    idx = close.index
    valid = ma_short.notna() & ma_long.notna()
    death = ma_short < ma_long
    up = valid & ~death & (close > ma_short)
    down = valid & (death | (close < ma_long))
    reg = pd.Series("neutral", index=idx, dtype=object)
    reg[up] = "accel_up"
    reg[down] = "accel_down"
    reg[~valid] = None
    return reg.dropna()


def _apply_confirmation(regime: pd.Series, confirm_days: int) -> pd.Series:
    """새 레짐이 confirm_days 거래일 연속으로 나올 때만 전환한다(그 전엔 직전 확정 상태 유지).

    잦은 뒤집힘(휩소)을 줄이는 대신 전환이 다소 늦어진다.
    """
    if regime.empty or confirm_days <= 1:
        return regime
    values = list(regime.values)
    committed = values[0]
    run_state = values[0]
    run_len = 1
    out = [committed]
    for value in values[1:]:
        if value == run_state:
            run_len += 1
        else:
            run_state = value
            run_len = 1
        if run_len >= confirm_days:
            committed = run_state
        out.append(committed)
    return pd.Series(out, index=regime.index)


def compute_ma_cross_regime(df: pd.DataFrame, confirm_days: int) -> pd.Series:
    """MA20/60 교차 + N일 확인 레짐을 계산한다.

    ``confirm_days`` 는 운영 설정의 N이다. N=0 은 원본 레짐 즉시 전환,
    N>0 은 새 레짐이 (N+1)거래일 연속 나올 때 확정 전환한다.
    """
    if confirm_days < 0:
        raise ValueError(f"confirm_days 는 0 이상이어야 합니다: {confirm_days}")
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=object)
    close = df["Close"].dropna().sort_index()
    if close.empty:
        return pd.Series(dtype=object)
    ma_short = close.rolling(MARKET_TREND_REGIME_MA_SHORT).mean()
    ma_long = close.rolling(MARKET_TREND_REGIME_MA_LONG).mean()
    raw = _regime_series_ma_cross(close, ma_short, ma_long)
    if confirm_days == 0:
        return raw
    return _apply_confirmation(raw, confirm_days + 1)


def _build_regime_ranges_from_series(regime: pd.Series, window_days: int) -> list[dict[str, Any]]:
    """레짐 시리즈를 최근 window_days 기준 연속 구간 목록으로 변환한다."""
    if regime is None or regime.empty:
        return []
    recent = regime.dropna().tail(int(window_days))
    ranges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for date_value, regime_value in recent.items():
        date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        if current is None or current["regime"] != regime_value:
            if current:
                ranges.append(current)
            current = {
                "regime": regime_value,
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


def _regime_backtest_metrics(
    regime: pd.Series, close: pd.Series, window: int, pos_map: dict[str, float]
) -> dict[str, Any]:
    """레짐 시리즈의 분포·휩소·유지기간 + 현금제어 전략 성과(vs buy&hold)를 계산한다.

    pos_map: 레짐별 투자비중(=1-현금%). 예: {accel_up:1.0, neutral:0.85, accel_down:0.7}.
    """
    reg = regime.tail(window)
    reg = reg[reg.notna()]
    if len(reg) < 2:
        return {}
    counts = reg.value_counts().to_dict()
    flips = int((reg.values[1:] != reg.values[:-1]).sum())
    runs, cur = [], 1
    for i in range(1, len(reg)):
        if reg.values[i] == reg.values[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    dwell = float(np.mean(runs)) if runs else 0.0

    # 룩어헤드 방지: 오늘 레짐으로 정한 포지션을 다음날 수익에 적용
    ret = np.log(close / close.shift(1)).reindex(reg.index)
    pos = reg.map(pos_map).shift(1)
    strat = (pos * ret).dropna()
    bh = ret.reindex(strat.index)

    def perf(series: pd.Series) -> tuple[float, float, float]:
        if series.empty or series.std() == 0:
            return 0.0, 0.0, 0.0
        cum = float(np.expm1(series.sum()) * 100)
        eq = np.exp(series.cumsum())
        mdd = float((eq / eq.cummax() - 1).min() * 100)
        sharpe = float(series.mean() / series.std() * np.sqrt(252))
        return cum, mdd, sharpe

    s_cum, s_mdd, s_sharpe = perf(strat)
    b_cum, b_mdd, b_sharpe = perf(bh)
    return {
        "up": int(counts.get("accel_up", 0)),
        "neutral": int(counts.get("neutral", 0)),
        "down": int(counts.get("accel_down", 0)),
        "flips": flips,
        "dwell": round(dwell, 1),
        "strat_return": round(s_cum, 1),
        "strat_mdd": round(s_mdd, 1),
        "strat_sharpe": round(s_sharpe, 2),
        "bh_return": round(b_cum, 1),
        "bh_mdd": round(b_mdd, 1),
        "bh_sharpe": round(b_sharpe, 2),
    }


def compute_regime_confirm_backtest(
    ticker: str | None = None,
    months: int = 12,
    up_cash: float | None = None,
    neutral_cash: float | None = None,
    down_cash: float | None = None,
) -> dict[str, Any]:
    """선택 지수(미지정이면 전체)의 MA20/60 교차 확인일수 후보를 최근 N개월로 비교한다.

    months: 백테스트 기간(개월, 1~36). up/neutral/down_cash: 레짐별 현금 비중(%).
    읽기 전용 분석 — config/DB 를 바꾸지 않는다.
    """
    if ticker is not None and not is_market_trend_index(ticker):
        allowed = ", ".join(idx["yf_ticker"] for idx in INDICES)
        raise ValueError(f"시장추세 지수({allowed}) 중 하나여야 합니다: {ticker}")
    months = int(months)
    if not (1 <= months <= 36):
        raise ValueError(f"백테스트 기간은 1~36개월이어야 합니다: {months}")
    window = max(10, months * TRADING_DAYS_PER_MONTH)

    cash = {
        "up": REGIME_BACKTEST_DEFAULT_CASH["up"] if up_cash is None else float(up_cash),
        "neutral": REGIME_BACKTEST_DEFAULT_CASH["neutral"] if neutral_cash is None else float(neutral_cash),
        "down": REGIME_BACKTEST_DEFAULT_CASH["down"] if down_cash is None else float(down_cash),
    }
    for label, value in cash.items():
        if not (0.0 <= value <= 100.0):
            raise ValueError(f"{label} 현금 비중은 0~100%여야 합니다: {value}")
    pos_map = {
        "accel_up": 1.0 - cash["up"] / 100.0,
        "neutral": 1.0 - cash["neutral"] / 100.0,
        "accel_down": 1.0 - cash["down"] / 100.0,
    }

    targets = [idx for idx in INDICES if ticker is None or idx["yf_ticker"] == ticker]

    indices_out: list[dict[str, Any]] = []
    for idx in targets:
        idx_ticker, name = idx["yf_ticker"], idx["name"]
        df = load_index_ohlc(idx_ticker)
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["Close"]).sort_index()
        close = df["Close"]

        # MA20/60 교차 방식 — 확인 필터 0~N일 나열
        ma_short = close.rolling(REGIME_BACKTEST_MA_SHORT).mean()
        ma_long = close.rolling(REGIME_BACKTEST_MA_LONG).mean()
        ma_raw = _regime_series_ma_cross(close, ma_short, ma_long)
        ma_variants: list[dict[str, Any]] = []
        for n in range(REGIME_BACKTEST_CONFIRM_MAX + 1):
            # n 일 확인 = 새 레짐이 (n+1)거래일 연속일 때 전환. n=0 은 즉시 전환(원본).
            conf = ma_raw if n == 0 else _apply_confirmation(ma_raw, n + 1)
            metrics = _regime_backtest_metrics(conf, close, window, pos_map)
            if metrics:
                ma_variants.append({"confirm_days": n, **metrics})
        if not ma_variants:
            continue

        indices_out.append(
            {
                "ticker": idx_ticker,
                "name": name,
                "confirm_days": _resolve_confirm_days(idx_ticker),
                "ma_variants": ma_variants,
                "ma_periods": [REGIME_BACKTEST_MA_SHORT, REGIME_BACKTEST_MA_LONG],
            }
        )
    return {
        "window_days": window,
        "months": months,
        "cash": cash,
        "indices": indices_out,
    }


def load_index_ohlc(yf_ticker: str) -> pd.DataFrame | None:
    """단일 지수의 일별 OHLC 히스토리를 반환한다 (한국: 네이버 5년, 미국: yfinance 10년).

    compute_index_history 와 탑픽 시장 레짐 계산이 공유하는 단일 소스.
    """
    index_meta = next((idx for idx in INDICES if idx["yf_ticker"] == yf_ticker), None)
    naver_symbol = (index_meta or {}).get("kor_naver_symbol")

    if naver_symbol:
        # 한국 인덱스: 네이버 차트에서 직접 받는다 (5년 ≈ 1250거래일, 여유 포함 1500).
        return _fetch_naver_kor_index_ohlc(naver_symbol, count=1500)

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
        return None

    if df is None or df.empty:
        return None

    # yfinance 가 단일 ticker 라도 컬럼을 멀티인덱스로 줄 수 있어 평탄화.
    cleaned_cols = {}
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        col_raw = df[col] if col in df.columns else None
        if col_raw is None:
            try:
                col_raw = df.xs(col, axis=1, level=0)
            except Exception:
                col_raw = None
        if col_raw is not None:
            if isinstance(col_raw, pd.DataFrame):
                col_raw = col_raw.iloc[:, 0]
            cleaned_cols[col] = col_raw
    return pd.DataFrame(cleaned_cols).dropna()


def _tail_streak(values: pd.Series) -> tuple[str | None, int]:
    """시리즈 마지막 값과 같은 값이 끝에서 몇 거래일 연속인지 반환한다."""
    cleaned = values.dropna()
    if cleaned.empty:
        return None, 0
    latest = cleaned.iloc[-1]
    streak = 0
    for value in reversed(cleaned.tolist()):
        if value != latest:
            break
        streak += 1
    return str(latest), streak


def _ma_cross_forecast(
    close: float | None,
    ma_short: float | None,
    ma_long: float | None,
    raw_regime: str | None,
    raw_streak: int,
    confirmed_regime: str | None,
    confirm_days: int,
) -> dict[str, Any] | None:
    """MA20/60 레짐 전환에 필요한 가격선과 남은 확인일수를 반환한다."""
    if close is None or close <= 0 or ma_short is None or ma_long is None or confirmed_regime is None:
        return None

    required_days = confirm_days + 1

    def change_pct(target_price: float | None) -> float | None:
        if target_price is None:
            return None
        return (target_price / close - 1.0) * 100.0

    def remaining(target_raw: str) -> int:
        if raw_regime == target_raw:
            return max(0, required_days - raw_streak)
        return required_days

    return {
        "confirm_days": confirm_days,
        "required_days": required_days,
        "raw_regime": raw_regime,
        "raw_streak": raw_streak,
        "up_price": ma_short,
        "up_pct": change_pct(ma_short),
        "up_remaining_days": remaining("accel_up"),
        "dn_price": ma_long,
        "dn_pct": change_pct(ma_long),
        "dn_remaining_days": remaining("accel_down"),
    }


def compute_index_history(yf_ticker: str) -> dict[str, Any]:
    """단일 지수의 가격/MA20/MA60/확정 레짐 히스토리를 반환한다.

    레짐은 MA20/60 교차 원본 신호에 지수별 확인일수를 적용한 값이다.
    """
    index_meta = next((idx for idx in INDICES if idx["yf_ticker"] == yf_ticker), None)
    name = index_meta["name"] if index_meta else yf_ticker

    ma_short_days = MARKET_TREND_REGIME_MA_SHORT
    ma_long_days = MARKET_TREND_REGIME_MA_LONG
    confirm_days = _resolve_confirm_days(yf_ticker)

    empty_payload = {
        "ticker": yf_ticker,
        "name": name,
        "ma_days": ma_short_days,
        "ma_short_days": ma_short_days,
        "ma_long_days": ma_long_days,
        "confirm_days": confirm_days,
        "history": [],
        "trend_min_12m": None,
        "trend_max_12m": None,
    }

    df = load_index_ohlc(yf_ticker)
    if df is None or df.empty:
        return empty_payload

    close_series = df["Close"].dropna()
    # 표(_build_item)와 동일하게 intraday 보정 — 마지막 점/레짐이 일치하도록.
    close_series = _apply_intraday_boost(close_series, yf_ticker)

    # 보정된 close_series 를 df 에 다시 반영 (인덱스가 확장되었을 경우 대비 reindex 적용)
    df = df.reindex(close_series.index)
    df["Close"] = close_series
    df = df.ffill()

    if len(close_series) < 2:
        return empty_payload

    ma_short_series = close_series.rolling(ma_short_days).mean()
    ma_long_series = close_series.rolling(ma_long_days).mean()
    raw_regime_series = _regime_series_ma_cross(close_series, ma_short_series, ma_long_series)
    confirmed_regime_series = compute_ma_cross_regime(pd.DataFrame({"Close": close_series}), confirm_days)
    try:
        st_period, st_multiplier = _resolve_supertrend_params(yf_ticker)
        st_df = _calculate_supertrend(df, period=st_period, multiplier=st_multiplier)
    except Exception:
        logger.exception("차트 표시용 SuperTrend 계산 실패: %s", yf_ticker)
        st_df = None

    # 최근 5년치 = 약 1200 거래일 (프론트에서 1개월~5년 범위 선택 가능)
    tail = TRADING_DAYS_PER_MONTH * 12 * 5
    length = len(close_series)
    take = min(length, tail)

    # 전체 시리즈에 대한 일별 trend% 사전 계산 (인덱스 = 0..length-1).
    full_trend: list[float | None] = []
    for idx in range(length):
        c = _to_float(close_series.iloc[idx])
        m = _to_float(ma_short_series.iloc[idx])
        if c is None or m is None or m == 0:
            full_trend.append(None)
        else:
            full_trend.append((c / m - 1.0) * 100.0)

    start = length - take
    # 추세점수 정규화 앵커는 표(_build_item)와 동일하게 — 최신 시점 기준 트레일링 12개월
    # 퍼센타일(config) 을 쓴다. (이전엔 5년 min/max 라 표와 점수가 어긋났다.)
    score_window = TRADING_DAYS_PER_MONTH * 12
    anchor_window = [v for v in full_trend[max(0, length - score_window) : length] if v is not None]
    if anchor_window:
        anchor_series = pd.Series(anchor_window, dtype="float64")
        upper_q = MARKET_TREND_SCORE_ANCHOR_PERCENTILE / 100.0
        score_min = float(anchor_series.quantile(1.0 - upper_q))
        score_max = float(anchor_series.quantile(upper_q))
    else:
        score_min = None
        score_max = None

    history: list[dict[str, Any]] = []

    for idx in range(start, length):
        date_value = close_series.index[idx]
        date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        close = _to_float(close_series.iloc[idx])
        ma_short_v = _to_float(ma_short_series.iloc[idx])
        ma_long_v = _to_float(ma_long_series.iloc[idx])
        trend = full_trend[idx]
        trend_score = (
            _normalize_score(trend, score_min, score_max) if score_min is not None and score_max is not None else None
        )

        # ST(SuperTrend) 기반 2단계 레짐
        st_val = _to_float(st_df["supertrend"].iloc[idx]) if st_df is not None else None
        st_dir = None
        if st_df is not None and "direction" in st_df.columns:
            st_dir = int(st_df["direction"].iloc[idx])

        regime = "accel_up" if st_dir == 1 else "accel_down"
        
        point_forecast = {
            "confirm_days": 0,
            "required_days": 0,
            "raw_regime": regime,
            "raw_streak": 0,
            "up_price": st_val if regime == "accel_down" else None,
            "up_pct": ((st_val / close - 1.0) * 100.0) if regime == "accel_down" and st_val and close else None,
            "up_remaining_days": 0,
            "dn_price": st_val if regime == "accel_up" else None,
            "dn_pct": ((st_val / close - 1.0) * 100.0) if regime == "accel_up" and st_val and close else None,
            "dn_remaining_days": 0,
        }

        history.append(
            {
                "date": date_str,
                "open": _to_float(df["Open"].iloc[idx]),
                "high": _to_float(df["High"].iloc[idx]),
                "low": _to_float(df["Low"].iloc[idx]),
                "close": close,
                "volume": _to_float(df["Volume"].iloc[idx]),
                "ma": ma_short_v,
                "ma_long": ma_long_v,
                "trend_pct": trend,
                "trend_score": trend_score,
                "regime": regime,
                "forecast": point_forecast,
                "supertrend": st_val,
                "supertrend_dir": st_dir,
            }
        )

    return {
        "ticker": yf_ticker,
        "name": name,
        "ma_days": ma_short_days,
        "ma_short_days": ma_short_days,
        "ma_long_days": ma_long_days,
        "confirm_days": confirm_days,
        "history": history,
        "trend_min_12m": score_min,
        "trend_max_12m": score_max,
    }


__all__ = [
    "compute_market_trend",
    "compute_index_history",
    "compute_ma_cross_regime",
    "compute_regime_confirm_backtest",
    "_resolve_confirm_days",
    "INDICES",
]
