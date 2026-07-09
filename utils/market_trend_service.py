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

import pandas as pd
import yfinance as yf

from config import (
    ALLOWED_MA_TYPES,
    MARKET_TREND_REGIME_BUFFER_PCT,
    MARKET_TREND_REGIME_MA_TYPE,
    MARKET_TREND_REGIME_SHORT_MA_DAYS,
    MARKET_TREND_SCORE_ANCHOR_PERCENTILE,
    MARKET_TREND_SUPERTREND_MULTIPLIER,
    MARKET_TREND_SUPERTREND_PERIOD,
    TRADING_DAYS_PER_MONTH,
)
from utils.moving_averages import calculate_moving_average

# 필수 설정값 유효성 검사
if MARKET_TREND_REGIME_MA_TYPE not in ALLOWED_MA_TYPES:
    raise ValueError(
        f"MARKET_TREND_REGIME_MA_TYPE 은 {ALLOWED_MA_TYPES} 중 하나여야 합니다. 현재 값: {MARKET_TREND_REGIME_MA_TYPE}"
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

    MA는 SMA {MARKET_TREND_REGIME_SHORT_MA_DAYS}일 고정.

    Returns:
        ``{"ma_days", "items": [{
            name, ticker, price, change_pct, trend_pct, trend_score,
            pct_from_high, current_regime, current_regime_days,
        }, ...]}``
    """
    ma_days = MARKET_TREND_REGIME_SHORT_MA_DAYS

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
    # 종가뿐 아니라 고가/저가도 받아야 SuperTrend(레짐 판정의 주 신호)를 계산할 수 있다.
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
    kor_close = kor_ohlc["Close"] if (kor_ohlc is not None and "Close" in kor_ohlc.columns) else None
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

        # 미국 인덱스 daily 마지막 종가가 지연 누락되면 intraday 마감가로 보강 (한국=네이버 제외).
        close_series = _apply_intraday_boost(close_series, yf_ticker)

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
        ma_series = calculate_moving_average(close_series, ma_days, MARKET_TREND_REGIME_MA_TYPE)
    except Exception:
        logger.exception("MA 계산 실패: %s (%s, days=%d)", yf_ticker, MARKET_TREND_REGIME_MA_TYPE, ma_days)
        return base

    base["trend_pct"] = _trend_pct_at(close_series, ma_series, offset=0)

    # 52주(252거래일) 전고점 대비 등락률. 현재가가 고점이면 0, 아래면 음수.
    high_window = close_series.tail(TRADING_DAYS_PER_MONTH * 12 + 12)
    high_52w = _to_float(high_window.max()) if not high_window.empty else None
    if high_52w is not None and high_52w > 0 and latest_price is not None:
        base["pct_from_high"] = (latest_price / high_52w - 1.0) * 100.0

    # SuperTrend 계산 — 한국 인덱스는 네이버 OHLC, 미국 인덱스는 yfinance df 에서 OHLC 를 취한다.
    supertrend_dir_series = None
    try:
        ticker_df: pd.DataFrame | None = None
        if kor_ohlc is not None and not kor_ohlc.empty:
            ticker_df = kor_ohlc.reindex(close_series.index)
        elif df is not None and not df.empty:
            ticker_df = pd.DataFrame(index=close_series.index)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if (yf_ticker, col) in df.columns:
                    ticker_df[col] = df[(yf_ticker, col)]
                elif col in df.columns:
                    ticker_df[col] = df[col]
        if ticker_df is not None:
            ticker_df = ticker_df.dropna(subset=["High", "Low", "Close"])
        if ticker_df is not None and not ticker_df.empty:
            st_period, st_multiplier = _resolve_supertrend_params(yf_ticker)
            st_df = _calculate_supertrend(
                ticker_df,
                period=st_period,
                multiplier=st_multiplier,
            )
            if st_df is not None and "direction" in st_df.columns:
                supertrend_dir_series = st_df["direction"]
    except Exception:
        logger.exception("_build_item 내 SuperTrend 계산 실패: %s", yf_ticker)

    # 최근 12개월 일별 레짐을 계산해 연속 구간으로 그룹화 → 현재 레짐 + 지속일수.
    # 레짐 판정용 MA도 같은 SMA {ma_days}일이므로 ma_series를 재활용한다.
    short_ma_series = ma_series
    buffer_pct = _resolve_regime_buffer(yf_ticker)
    ranges = _build_daily_regime_ranges(
        close_series, ma_series, short_ma_series, buffer_pct, supertrend_dir_series
    )
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

        if ranges[-1]["regime"] != "accel_up":
            base["days_since_last_up"] = _days_since_last("accel_up")
        if ranges[-1]["regime"] == "accel_down":
            base["days_since_last_neutral"] = _days_since_last("neutral")

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


def _regime_step(
    close_raw: float | None,
    close_smooth: float | None,
    supertrend_dir: int | None,
    buffer_pct: float,
) -> str | None:
    """레짐 판정 — **슈퍼트렌드(ST) 방향이 주(主), MA±버퍼가 보조(강도 확인)**.

    ST가 방향(상승/하락)을 정하고, MA 는 방향을 뒤집지 못하며 '아직 확인 안 됨 → 중립'
    으로만 끌어내린다(ST 는 자체 추세추종이라 별도 히스테리시스 불필요).
    버퍼는 지수별로 다르다(``buffer_pct`` 로 주입).

    판정 기준(close_smooth = MA):
        ST ▲ 이고 종가 > MA × (1 + 버퍼) → 상승 (accel_up)
        ST ▼ 이고 종가 < MA × (1 − 버퍼) → 하락 (accel_down)
        그 외(ST·가격 방향 불일치, MA 근처 등) → 중립 (neutral)
    ST 방향이 없으면(데이터 부족) 레짐도 판정하지 않는다(None).
    """
    if close_raw is None or close_smooth is None or close_smooth <= 0 or supertrend_dir is None:
        return None

    buffer_ratio = buffer_pct / 100.0
    if supertrend_dir == 1 and close_raw > close_smooth * (1.0 + buffer_ratio):
        return "accel_up"
    if supertrend_dir == -1 and close_raw < close_smooth * (1.0 - buffer_ratio):
        return "accel_down"
    return "neutral"


def _resolve_supertrend_params(yf_ticker: str) -> tuple[int, float]:
    """지수(yf_ticker)별 슈퍼트렌드 (기간, 곱수)를 반환한다.

    기간은 전 지수 공통(MARKET_TREND_SUPERTREND_PERIOD), 곱수는 지수별로 등록된 값을 쓴다.
    등록되지 않은 지수는 설정 누락이므로 명시적으로 오류를 낸다.
    """
    if yf_ticker not in MARKET_TREND_SUPERTREND_MULTIPLIER:
        raise ValueError(
            f"MARKET_TREND_SUPERTREND_MULTIPLIER 에 지수 '{yf_ticker}' 의 곱수가 등록되지 않았습니다. "
            "config.py 에 해당 지수를 등록해주세요."
        )
    return MARKET_TREND_SUPERTREND_PERIOD, MARKET_TREND_SUPERTREND_MULTIPLIER[yf_ticker]


def _resolve_regime_buffer(yf_ticker: str) -> float:
    """지수(yf_ticker)별 레짐 판정 버퍼(%)를 반환한다.

    지수별로 등록된 값을 쓰며, 등록되지 않은 지수는 설정 누락이므로 명시적으로 오류를 낸다.
    """
    if yf_ticker not in MARKET_TREND_REGIME_BUFFER_PCT:
        raise ValueError(
            f"MARKET_TREND_REGIME_BUFFER_PCT 에 지수 '{yf_ticker}' 의 버퍼가 등록되지 않았습니다. "
            "config.py 에 해당 지수를 등록해주세요."
        )
    return MARKET_TREND_REGIME_BUFFER_PCT[yf_ticker]


def _calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """SuperTrend 지표 계산 (ATR 기반 상/하단 트렌드 밴드 및 방향 판정)."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's Smoothing ATR
    atr = tr.ewm(alpha=1.0/period, adjust=False).mean()
    
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1 = up, -1 = down
    
    for i in range(1, len(df)):
        prev_upper = final_upper.iloc[i-1]
        prev_lower = final_lower.iloc[i-1]
        prev_close = close.iloc[i-1]
        
        # Upper Band
        if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_upper
            
        # Lower Band
        if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_lower
            
        # Direction
        prev_dir = direction.iloc[i-1]
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
    
    return pd.DataFrame({
        'supertrend': supertrend,
        'direction': direction
    }, index=df.index)


def _build_daily_regime_ranges(
    close_series: pd.Series,
    ma_series: pd.Series,
    short_ma_series: pd.Series | None,
    buffer_pct: float,
    supertrend_dir_series: pd.Series | None = None,
    window_days: int = TRADING_DAYS_PER_MONTH * 12,
) -> list[dict[str, Any]]:
    """최근 ``window_days`` 거래일의 일별 레짐(ST 주도 + MA 보조)을 연속 구간으로 그룹화한다.

    레짐은 당일 (종가·MA·ST방향) 만의 함수라 상태 워밍업이 필요 없다.
    반환 형식: [{"regime": str, "start_date": str, "end_date": str, "days": int}, ...]
    오래된 순서 → 최신 순서.
    """
    if close_series is None or ma_series is None or short_ma_series is None:
        return []
    length = min(len(close_series), len(ma_series), len(short_ma_series))
    if length == 0:
        return []

    # 휩소 방지를 위해 당일 종가 대신 이동평균을 레짐 판정용 비교값으로 사용합니다.
    try:
        close_smooth = calculate_moving_average(
            close_series, MARKET_TREND_REGIME_SHORT_MA_DAYS, MARKET_TREND_REGIME_MA_TYPE
        ).fillna(close_series)
    except Exception:
        close_smooth = close_series

    take = min(length, int(window_days))
    start_idx = length - take
    ranges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    # 슈퍼트렌드 방향 매핑 맵
    st_dir_map = {}
    if supertrend_dir_series is not None:
        st_dir_map = supertrend_dir_series.dropna().to_dict()

    for idx in range(start_idx, length):  # 레짐은 당일 (종가·MA·ST방향) 만의 함수 → 워밍업 불필요
        date_value = close_series.index[idx]
        st_dir = st_dir_map.get(date_value)
        if st_dir is not None:
            st_dir = int(st_dir)

        regime = _regime_step(
            _to_float(close_series.iloc[idx]),
            _to_float(close_smooth.iloc[idx]),
            st_dir,
            buffer_pct,
        )

        date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
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


def _ma_newest_weight(close_series: pd.Series, ma_days: int) -> float | None:
    """이동평균에서 '가장 최근 종가 1단위'가 MA 마지막 값에 기여하는 가중치 w.

    이동평균은 선형 시불변이라 이 가중치는 날짜·값과 무관한 상수다.
    끝점을 미세 섭동해 수치적으로 한 번만 구한다. 이 w 로
    가상의 다음날 종가 P 에 대한 다음날 MA 를 ``w*P + R`` 로 O(1) 평가할 수 있다.
    """
    if close_series is None or len(close_series) < ma_days + 1:
        return None
    try:
        base_last = _to_float(calculate_moving_average(close_series, ma_days, MARKET_TREND_REGIME_MA_TYPE).iloc[-1])
        if base_last is None:
            return None
        delta = abs(_to_float(close_series.iloc[-1]) or 1.0) * 0.01 + 1.0
        bumped = close_series.copy()
        bumped.iloc[-1] = float(bumped.iloc[-1]) + delta
        bumped_last = _to_float(calculate_moving_average(bumped, ma_days, MARKET_TREND_REGIME_MA_TYPE).iloc[-1])
        if bumped_last is None:
            return None
    except Exception:
        return None
    return (bumped_last - base_last) / delta


def _forecast_thresholds(
    current_close: float,
    w_long: float,
    r_long: float,
    w_short: float,
    r_short: float,
    st_line: float,
    buffer_pct: float,
) -> dict[str, Any] | None:
    """'다음 영업일 종가가 현재 대비 몇 %'일 때 레짐이 바뀌는 두 경계를 산출 (레벨 기반).

    다음날 MA 를 ``w*P + r`` (LTI 선형)로 O(1) 평가한다. 내일 ST 방향은 내일 종가 P 가
    현재 ST 밴드선(st_line)을 넘는지로 결정된다(P > st_line → 상승, 그 외 → 하락).
    레짐은 (종가·MA·ST방향) 만의 함수라 등락률→레짐이 단조 → 이진 탐색.

    Returns:
        ``{"up_pct","up_price","dn_pct","dn_price"}`` (상승↔중립=up, 중립↔하락=dn).
        탐색 범위에서 경계를 못 찾으면 해당 값은 None. 둘 다 None 이면 None.
    """
    if current_close <= 0:
        return None
    rank = {"accel_up": 2, "neutral": 1, "accel_down": 0}

    def regime_for(pct: float) -> str | None:
        price = current_close * (1.0 + pct / 100.0)
        if price <= 0:
            return None
        sm = w_short * price + r_short  # 내일 MA
        st_dir_next = 1 if price > st_line else -1  # 내일 ST 방향 (밴드선 돌파 여부)
        return _regime_step(price, sm, st_dir_next, buffer_pct)

    pct_hi, pct_lo = 90.0, -90.0  # 내일 단일일 등락률 탐색 범위(상승/하락 경계 대칭)

    def boundary(min_rank: int) -> float | None:
        r_hi, r_lo = regime_for(pct_hi), regime_for(pct_lo)
        if r_hi is None or r_lo is None:
            return None
        if (rank[r_hi] >= min_rank) == (rank[r_lo] >= min_rank):
            return None
        low, high = pct_lo, pct_hi
        for _ in range(24):
            mid = (low + high) / 2.0
            r = regime_for(mid)
            if r is not None and rank[r] >= min_rank:
                high = mid
            else:
                low = mid
        return round((low + high) / 2.0, 2)

    t_up = boundary(2)
    t_dn = boundary(1)

    def price_at(pct: float | None) -> float | None:
        return round(current_close * (1.0 + pct / 100.0), 2) if pct is not None else None

    result: dict[str, Any] = {
        "up_pct": t_up,
        "up_price": price_at(t_up),
        "dn_pct": t_dn,
        "dn_price": price_at(t_dn),
    }

    if result["up_pct"] is None and result["dn_pct"] is None:
        return None
    return result


def compute_index_history(yf_ticker: str) -> dict[str, Any]:
    """단일 지수의 최근 12개월 가격/추세 히스토리 + 각 일자별 레짐을 반환한다 (행 펼침용).

    MA는 SMA {MARKET_TREND_REGIME_SHORT_MA_DAYS}일 고정.
    각 history 항목의 ``forecast`` 는 그 일자 기준 '내일 종가 전환 예측'
    ``{up_pct, up_price, dn_pct, dn_price}`` (없으면 None) 이다.

    Returns:
        ``{"ticker", "name", "ma_days",
            "history": [{date, close, ma, trend_pct, trend_score, regime, forecast}, ...],
            "trend_min_12m", "trend_max_12m"}``
        해당 ticker 가 알려진 인덱스가 아니면 name 은 ticker 그대로 사용.
    """
    index_meta = next((idx for idx in INDICES if idx["yf_ticker"] == yf_ticker), None)
    name = index_meta["name"] if index_meta else yf_ticker
    naver_symbol = (index_meta or {}).get("kor_naver_symbol")

    ma_days = MARKET_TREND_REGIME_SHORT_MA_DAYS
    buffer_pct = _resolve_regime_buffer(yf_ticker)

    empty_payload = {
        "ticker": yf_ticker,
        "name": name,
        "ma_days": ma_days,
        "buffer_pct": buffer_pct,
        "history": [],
        "trend_min_12m": None,
        "trend_max_12m": None,
    }

    df: pd.DataFrame | None = None
    if naver_symbol:
        # 한국 인덱스: 네이버 차트에서 직접 받는다 (5년 ≈ 1250거래일, 여유 포함 1500).
        df = _fetch_naver_kor_index_ohlc(naver_symbol, count=1500)
        if df is None:
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
        df = pd.DataFrame(cleaned_cols).dropna()

    close_series = df["Close"].dropna()
    # 표(_build_item)와 동일하게 intraday 보정 — 마지막 점/레짐이 일치하도록.
    close_series = _apply_intraday_boost(close_series, yf_ticker)

    # 보정된 close_series 를 df 에 다시 반영 (인덱스가 확장되었을 경우 대비 reindex 적용)
    df = df.reindex(close_series.index)
    df["Close"] = close_series
    df = df.ffill()

    # SuperTrend 계산 (지수별 파라미터 적용)
    try:
        st_period, st_multiplier = _resolve_supertrend_params(yf_ticker)
        st_df = _calculate_supertrend(
            df,
            period=st_period,
            multiplier=st_multiplier,
        )
    except Exception:
        logger.exception("SuperTrend 계산 실패: %s", yf_ticker)
        st_df = None

    if len(close_series) < 2:
        return empty_payload

    try:
        ma_series = calculate_moving_average(close_series, ma_days, MARKET_TREND_REGIME_MA_TYPE)
    except Exception:
        logger.exception("MA 계산 실패: %s (%s, days=%d)", yf_ticker, MARKET_TREND_REGIME_MA_TYPE, ma_days)
        ma_series = None
    # 레짐 판정용 MA도 같은 MA_TYPE {ma_days}일이므로 ma_series를 재활용한다.
    short_ma_series = ma_series

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
    anchor_window = [v for v in full_trend[max(0, length - score_window) : length] if v is not None]
    if anchor_window:
        anchor_series = pd.Series(anchor_window, dtype="float64")
        upper_q = MARKET_TREND_SCORE_ANCHOR_PERCENTILE / 100.0
        score_min = float(anchor_series.quantile(1.0 - upper_q))
        score_max = float(anchor_series.quantile(upper_q))
    else:
        score_min = None
        score_max = None

    # 일자별 '내일 종가 기준 전환 예측'을 위해 MA 최신가중치(LTI 상수)를 한 번만 구한다.
    # 다음날 MA = weight*P + R 로 O(1) 평가 → 날짜마다 MA 재계산 없이 forecast 산출.
    # 장기/단기 MA가 동일(SMA {ma_days}일)하므로 가중치도 하나.
    ma_weight = _ma_newest_weight(close_series, ma_days) if ma_series is not None else None
    short_weight = ma_weight
    # 마지막 일자는 '내일'이 없으므로 가상 평탄일 1개를 붙여 R 을 한 번만 계산한다.
    r_last: float | None = None
    r_last_short: float | None = None
    if ma_weight is not None and short_weight is not None and length >= 1:
        last_close = _to_float(close_series.iloc[length - 1])
        if last_close is not None:
            try:
                ext = pd.concat(
                    [close_series, pd.Series([last_close], index=[close_series.index[-1] + pd.Timedelta(days=1)])]
                )
                ext_last = _to_float(calculate_moving_average(ext, ma_days, MARKET_TREND_REGIME_MA_TYPE).iloc[-1])
                if ext_last is not None:
                    r_last = ext_last - ma_weight * last_close
                    r_last_short = r_last  # 장기/단기 MA 동일
            except Exception:
                r_last = None
                r_last_short = None

    # 휩소 방지를 위해 당일 종가 대신 이동평균을 레짐 판정용 비교값으로 사용합니다.
    try:
        close_smooth = calculate_moving_average(
            close_series, MARKET_TREND_REGIME_SHORT_MA_DAYS, MARKET_TREND_REGIME_MA_TYPE
        ).fillna(close_series)
    except Exception:
        close_smooth = close_series

    history: list[dict[str, Any]] = []

    for idx in range(start, length):
        date_value = close_series.index[idx]
        date_str = date_value.strftime("%Y-%m-%d") if hasattr(date_value, "strftime") else str(date_value)
        close = _to_float(close_series.iloc[idx])
        ma_v = _to_float(ma_series.iloc[idx]) if ma_series is not None else None
        trend = full_trend[idx]
        trend_score = (
            _normalize_score(trend, score_min, score_max) if score_min is not None and score_max is not None else None
        )

        st_val = _to_float(st_df["supertrend"].iloc[idx]) if st_df is not None else None
        st_dir = None
        if st_df is not None and "direction" in st_df.columns:
            st_dir = int(st_df["direction"].iloc[idx])

        # 레짐: ST 방향(주) + MA±버퍼(보조).
        regime = _regime_step(close, _to_float(close_smooth.iloc[idx]), st_dir, buffer_pct)

        # 그 시점 기준 '다음 영업일' 전환 예측 (과거 일자엔 그날의 예측이 그대로 보존된다).
        # 내일 ST 방향은 '내일 종가 vs 현재 ST 밴드선(st_val)' 으로 결정된다.
        point_forecast: dict[str, Any] | None = None
        if ma_weight is not None and short_weight is not None and close is not None and st_val is not None:
            if idx < length - 1:
                c_next = _to_float(close_series.iloc[idx + 1])
                m_next = _to_float(ma_series.iloc[idx + 1]) if ma_series is not None else None
                sm_next = _to_float(short_ma_series.iloc[idx + 1]) if short_ma_series is not None else None
                r_next = (m_next - ma_weight * c_next) if (c_next is not None and m_next is not None) else None
                r_next_short = (
                    (sm_next - short_weight * c_next) if (c_next is not None and sm_next is not None) else None
                )
            else:
                r_next = r_last
                r_next_short = r_last_short
            if r_next is not None and r_next_short is not None:
                point_forecast = _forecast_thresholds(
                    close, ma_weight, r_next, short_weight, r_next_short, st_val, buffer_pct
                )

        history.append(
            {
                "date": date_str,
                "open": _to_float(df["Open"].iloc[idx]),
                "high": _to_float(df["High"].iloc[idx]),
                "low": _to_float(df["Low"].iloc[idx]),
                "close": close,
                "volume": _to_float(df["Volume"].iloc[idx]),
                "ma": ma_v,
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
        "ma_days": ma_days,
        "buffer_pct": buffer_pct,
        "history": history,
        "trend_min_12m": score_min,
        "trend_max_12m": score_max,
    }


__all__ = ["compute_market_trend", "compute_index_history", "INDICES"]
