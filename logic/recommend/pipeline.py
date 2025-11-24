"""국가 기반 간소화 추천 파이프라인."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

import config

# 데이터 디렉토리 경로 설정
STOCKS_DIR = Path(__file__).resolve().parents[2] / "zsettings" / "stocks"
from utils.settings_loader import (
    AccountSettingsError,
    get_account_settings,
    get_strategy_rules,
    load_common_settings,
    resolve_strategy_params,
)
from strategies.maps.constants import DECISION_CONFIG, DECISION_MESSAGES, DECISION_NOTES
from logic.common import (
    sort_decisions_by_order_and_score,
    filter_category_duplicates,
    get_buy_signal_streak,
    is_category_exception,
)
from strategies.maps.history import (
    calculate_consecutive_holding_info,
    calculate_trade_cooldown_info,
)
from utils.stock_list_io import get_etfs
from utils.data_loader import (
    prepare_price_data,
    get_latest_trading_day,
    get_next_trading_day,
    count_trading_days,
    fetch_naver_etf_inav_snapshot,
    get_trading_days,
    MissingPriceDataError,
)
from utils.indicators import calculate_ma_score
from utils.moving_averages import calculate_moving_average
from strategies.rsi.recommend import calculate_rsi_for_ticker
from utils.db_manager import get_db_connection, list_open_positions
from utils.logger import get_app_logger
from utils.market_schedule import get_market_open_time

logger = get_app_logger()

# 거래일 캐시 워밍업 플래그 (전역 변수, 한 번만 실행)
_trading_days_cache_warmed_up = set()


def _warmup_trading_days_cache(country_code: str) -> None:
    """거래일 캐시를 미리 로딩하여 성능 향상 (국가별 1회만 실행)"""
    if country_code in _trading_days_cache_warmed_up:
        return

    try:
        warmup_start = time.perf_counter()
        # 과거 5년 + 미래 2년치 거래일 미리 조회
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
        end_date = (pd.Timestamp.now() + pd.DateOffset(years=2)).strftime("%Y-%m-%d")

        trading_days = get_trading_days(start_date, end_date, country_code)
        _trading_days_cache_warmed_up.add(country_code)

        logger.info(
            "[%s] 거래일 캐시 워밍업 완료 (%.2fs, %d개 거래일)",
            country_code.upper(),
            time.perf_counter() - warmup_start,
            len(trading_days),
        )
    except Exception as e:
        logger.warning("[%s] 거래일 캐시 워밍업 실패: %s", country_code.upper(), e)


@dataclass
class RecommendationReport:
    account_id: str
    country_code: str
    base_date: pd.Timestamp
    recommendations: List[Dict[str, Any]]
    report_date: datetime
    summary_data: Optional[Dict[str, Any]] = None
    header_line: Optional[str] = None
    detail_headers: Optional[List[str]] = None
    detail_rows: Optional[List[List[Any]]] = None
    detail_extra_lines: Optional[List[str]] = None
    decision_config: Dict[str, Any] = None


@dataclass
class _TickerMeta:
    ticker: str
    name: str
    category: str


@dataclass
class _TickerScore:
    meta: _TickerMeta
    price: float
    prev_close: float
    daily_pct: float
    score: float
    streak: int
    category: str
    ma_value: float = 0.0


def _load_full_etf_meta(country_code: str) -> Dict[str, Dict[str, Any]]:
    """Load metadata for all ETFs including recommend_disabled ones."""

    file_path = STOCKS_DIR / f"{country_code}.json"
    if not file_path.exists():
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("전체 ETF 메타 로드 실패: %s", exc)
        return {}

    meta_map: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, list):
        return meta_map

    for block in data:
        if not isinstance(block, dict):
            continue

        raw_category = block.get("category")
        if isinstance(raw_category, (list, set, tuple)):
            raw_category = next(iter(raw_category), "") if raw_category else ""
        category_name = str(raw_category or "").strip()
        if not category_name:
            raise ValueError(f"카테고리 블록에 카테고리 이름이 없습니다. 모든 카테고리 블록은 'category' 필드가 있어야 합니다.")

        tickers = block.get("tickers") or []
        if not isinstance(tickers, list):
            continue

        for item in tickers:
            if not isinstance(item, dict):
                continue

            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker:
                continue

            raw_item_category = item.get("category", category_name)
            if isinstance(raw_item_category, (list, set, tuple)):
                raw_item_category = next(iter(raw_item_category), "") if raw_item_category else ""
            item_category = str(raw_item_category or category_name or "").strip()
            if not item_category:
                raise ValueError(f"종목 {ticker}의 카테고리가 없습니다. 모든 종목은 카테고리가 있어야 합니다.")

            name = str(item.get("name") or ticker).strip() or ticker

            meta_map[ticker] = {
                "ticker": ticker,
                "name": name,
                "category": item_category,
            }

    return meta_map


def _fetch_dataframe(
    ticker: str,
    *,
    country: str,
    ma_period: int,
    base_date: Optional[pd.Timestamp],
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    data_window: Optional[tuple[str, str]] = None,
) -> Optional[pd.DataFrame]:
    window_start: Optional[str]
    window_end: Optional[str]
    if data_window:
        window_start, window_end = data_window
    else:
        window_start = None
        window_end = None
    try:
        if prefetched_data and ticker in prefetched_data:
            df = prefetched_data[ticker]
        else:
            raise MissingPriceDataError(
                country=country,
                start_date=window_start,
                end_date=window_end or (base_date.strftime("%Y-%m-%d") if base_date is not None else None),
                tickers=[ticker],
            )

        if df is None or df.empty:
            logger.warning("%s에 대한 데이터를 가져오지 못했습니다.", ticker)
            return None

        if "Close" not in df.columns:
            logger.warning("%s에 대한 종가(Close) 데이터가 없습니다.", ticker)
            return None

        df = df.dropna(subset=["Close"])

        if df.empty:
            logger.warning("%s에 대한 유효한 데이터가 없습니다.", ticker)
            return None

        return df

    except Exception as e:
        logger.warning("%s 데이터 처리 중 오류 발생: %s", ticker, e)
        import traceback

        traceback.print_exc()
        return None


def _calc_metrics(df: pd.DataFrame, ma_period: int) -> Optional[tuple]:
    try:
        # 'Close' 또는 'Adj Close' 중 사용 가능한 컬럼 선택
        if "unadjusted_close" in df.columns:
            raw_close = df["unadjusted_close"].astype(float)
        else:
            raw_close = df["Close"].astype(float)

        if "Adj Close" in df.columns and not df["Adj Close"].isnull().all():
            price_series = df["Adj Close"].astype(float)
        else:
            price_series = raw_close

        raw_close = raw_close.dropna()
        price_series = price_series.dropna()

        if raw_close.empty or price_series.empty:
            return None

        # 이동평균 계산 (최소 1개 데이터로도 계산 가능하도록 min_periods=1 설정)
        ma = price_series.rolling(window=ma_period, min_periods=1).mean()

        latest_close = float(price_series.iloc[-1])
        prev_close = float(price_series.iloc[-2]) if len(price_series) >= 2 else latest_close
        ma_value = float(ma.iloc[-1]) if not pd.isna(ma.iloc[-1]) else latest_close

        # 0 이하 값이면 기본값 설정
        if ma_value <= 0:
            ma_value = latest_close if latest_close > 0 else 1.0

        if latest_close <= 0:
            return None

        # 일간 수익률 계산 (이전 종가가 없거나 0 이하면 0%로 처리)
        daily_pct = 0.0
        raw_prev_close = float(raw_close.iloc[-2]) if len(raw_close) >= 2 else float(raw_close.iloc[-1])
        raw_latest_close = float(raw_close.iloc[-1])
        if raw_prev_close and raw_prev_close > 0:
            daily_pct = ((raw_latest_close / raw_prev_close) - 1.0) * 100

        # 점수 계산 (이동평균 대비 수익률, % 단위)
        score = 0.0
        if ma_value > 0:
            score = ((latest_close / ma_value) - 1.0) * 100

        # 점수가 매우 작으면 0.01%로 처리 (0점 방지)
        if abs(score) < 0.01 and score != 0:
            score = 0.01 if score > 0 else -0.01

        # 연속 상승일 계산
        streak = 0
        for price, ma_entry in zip(reversed(price_series.iloc[-ma_period:]), reversed(ma.iloc[-ma_period:])):
            if pd.isna(ma_entry) or pd.isna(price) or price < ma_entry:
                break
            streak += 1

        # 보유일 계산 (최대 20일로 제한)
        holding_days = min(streak, 20) if streak > 0 else 0

        return latest_close, prev_close, daily_pct, score, holding_days
    except Exception as e:
        logger.exception("메트릭 계산 중 오류 발생: %s", e)
        return None


def _select_price_series(df: pd.DataFrame, country_code: str) -> pd.Series:
    """Close 가격 시리즈를 반환합니다."""
    close_series = pd.to_numeric(df.get("Close"), errors="coerce") if "Close" in df.columns else None

    if close_series is None:
        raise ValueError("가격 시리즈를 찾을 수 없습니다 (Close 열 없음).")

    return close_series.fillna(method="ffill").fillna(method="bfill")


def _build_score(meta: _TickerMeta, metrics) -> _TickerScore:
    # 메트릭이 없는 경우 기본값 반환
    if metrics is None:
        logger.warning("%s에 대한 메트릭이 없습니다.", meta.ticker)
        return _TickerScore(
            meta=meta,
            price=0.0,
            prev_close=0.0,
            daily_pct=0.0,
            score=0.0,
            streak=0,
            category="",  # 카테고리를 빈 문자열로 초기화
        )

    try:
        price, prev_close, daily_pct, score, holding_days = metrics

        # 점수가 매우 작은 경우 0으로 처리하지 않도록 합니다.
        if abs(score) < 0.01 and score != 0:
            score = 0.01 if score > 0 else -0.01

        # 카테고리 정보가 없는 경우 빈 문자열로 설정
        category = ""
        if hasattr(meta, "category") and meta.category is not None:
            # 카테고리가 리스트인 경우 첫 번째 항목을 사용
            if isinstance(meta.category, (set, list)):
                category = str(next(iter(meta.category), "")) if meta.category else ""
            else:
                category = str(meta.category)

        # 점수가 None이 아니면 그대로 사용, None이면 0.0으로 설정
        final_score = float(round(score, 2)) if score is not None else 0.0

        # 이동평균 값 계산 (점수를 이용해서 근사치 계산)
        ma_value = price * (1 - score / 100) if price > 0 else 0.0

        return _TickerScore(
            meta=meta,
            price=float(price) if price is not None else 0.0,
            prev_close=float(prev_close) if prev_close is not None else 0.0,
            daily_pct=float(round(daily_pct, 2)) if daily_pct is not None else 0.0,
            score=final_score,
            streak=int(holding_days) if holding_days is not None else 0,
            category=category.strip(),  # 공백 제거
            ma_value=ma_value,
        )
    except Exception as e:
        logger.exception("%s 점수 생성 중 오류 발생: %s", meta.ticker, e)
        # 오류 발생 시 기본값 반환
        return _TickerScore(
            meta=meta,
            price=0.0,
            prev_close=0.0,
            daily_pct=0.0,
            score=0.0,
            streak=0,
            category=meta.category if hasattr(meta, "category") else "",
        )


def _resolve_base_date(account_id: str, date_str: Optional[str]) -> pd.Timestamp:
    if date_str:
        try:
            base = pd.to_datetime(date_str).normalize()
        except Exception as exc:
            raise ValueError(f"잘못된 날짜 형식입니다: {date_str}") from exc
    else:
        account_settings = get_account_settings(account_id)
        country_code = (account_settings.get("country_code") or account_id).strip().lower()
        latest_trading_day = get_latest_trading_day(country_code)
        base = latest_trading_day.normalize()

    return base.normalize()


def _apply_precision(value: float, precision: int) -> float | int:
    if precision <= 0:
        return int(round(value))
    return round(value, precision)


def _resolve_state_phrase(state: str) -> str:
    state_key = (state or "").upper()
    if state_key == "BUY":
        return DECISION_MESSAGES.get("NEW_BUY", "")
    return ""


def _resolve_state_order(state: str) -> int:
    state_key = (state or "").upper()
    cfg = DECISION_CONFIG.get(state_key, {})
    return int(cfg.get("order", 99))


def _join_phrase_parts(*parts: Optional[str]) -> str:
    """Join non-empty phrase components with a separator."""

    cleaned: List[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            cleaned.append(text)
    return " | ".join(cleaned)


def _format_sell_replace_phrase(phrase: str, *, etf_meta: Dict[str, Dict[str, Any]]) -> str:
    if not phrase or "교체매도" not in phrase:
        return phrase

    ratio_match = re.search(r"손익률\s+[+-]?[0-9.,]+%", phrase)
    ticker_matches = re.findall(r"([A-Za-z0-9:]+)\(으\)로 교체", phrase)

    if not ratio_match or not ticker_matches:
        return phrase

    target_ticker = ticker_matches[-1]
    ratio_text = ratio_match.group(0)
    target_meta = etf_meta.get(target_ticker) or etf_meta.get(target_ticker.upper()) or {}
    target_name = target_meta.get("name") or target_ticker

    return f"교체매도 {ratio_text} - {target_name}({target_ticker})로 교체"


def _format_min_score_phrase(score_value: Optional[float], min_buy_score: float) -> str:
    template = DECISION_NOTES.get("MIN_SCORE", "최소 {min_buy_score:.1f}점수 미만")
    try:
        base = template.format(min_buy_score=min_buy_score)
    except Exception:
        base = f"최소 {min_buy_score:.1f}점수 미만"

    if score_value is None or pd.isna(score_value):
        return f"{base} (현재 점수 없음)"
    return f"{base} (현재 {score_value:.1f})"


def _normalize_buy_date(value: Any) -> Optional[pd.Timestamp]:
    """Convert various buy date formats into a normalized pandas Timestamp."""

    if value in (None, "", "-"):
        return None
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return None

    if pd.isna(ts):
        return None

    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert(None)  # type: ignore[attr-defined]
        except AttributeError:
            ts = ts.tz_localize(None)  # type: ignore[attr-defined]
    return ts.normalize()


def _resolve_buy_price(
    ticker_data: Dict[str, Any],
    buy_date: Optional[pd.Timestamp],
    *,
    fallback_price: Optional[float] = None,
) -> Optional[float]:
    """Pick the closest available closing price on or before the buy date."""

    if buy_date is None:
        return fallback_price

    close_series = ticker_data.get("close")
    if not isinstance(close_series, pd.Series) or close_series.empty:
        return fallback_price

    series = close_series.dropna().copy()
    if series.empty:
        return fallback_price

    try:
        index = pd.to_datetime(series.index)
    except Exception:
        return fallback_price

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)  # type: ignore[attr-defined]
    index = index.normalize()
    normalized_series = pd.Series(series.values, index=index)
    normalized_series = normalized_series.sort_index()
    normalized_series = normalized_series[~normalized_series.index.duplicated(keep="last")]

    if normalized_series.empty:
        return fallback_price

    prior_or_same = normalized_series.loc[normalized_series.index <= buy_date]
    if not prior_or_same.empty:
        return float(prior_or_same.iloc[-1])

    after = normalized_series.loc[normalized_series.index >= buy_date]
    if not after.empty:
        return float(after.iloc[0])

    return fallback_price


def _compute_trailing_return(
    close_series: pd.Series,
    periods_back: int,
) -> float:
    """Compute percentage return using close price N trading days ago."""

    if not isinstance(close_series, pd.Series) or close_series.empty:
        return 0.0
    valid = close_series.dropna()
    if valid.empty:
        return 0.0

    # 데이터가 충분하지 않으면 0.0 반환 (최소 periods_back + 1개 필요)
    if len(valid) < periods_back + 1:
        return 0.0

    try:
        latest_price = float(valid.iloc[-1])
        prev_price = float(valid.iloc[-(periods_back + 1)])
    except (IndexError, TypeError, ValueError):
        return 0.0

    if prev_price <= 0:
        return 0.0

    return round(((latest_price / prev_price) - 1.0) * 100.0, 2)


def _build_ticker_timeseries_entry(
    *,
    ticker: str,
    df: pd.DataFrame,
    country_code: str,
    base_date: pd.Timestamp,
    ma_period: int,
    ma_type: str,
    min_buy_score: float,
    realtime_inav_snapshot: Optional[Dict[str, Dict[str, float]]] = None,
) -> Optional[Dict[str, Any]]:
    """대표군 여부와 상관없이 동일한 방식으로 점수/지표를 계산합니다."""

    ticker_upper = str(ticker or "").strip().upper()
    if not ticker_upper or df is None or df.empty:
        return None

    df_sorted = df.sort_index()
    price_series = _select_price_series(df_sorted, country_code)
    if price_series is None or price_series.empty:
        return None
    price_series = price_series.dropna()
    if price_series.empty:
        return None

    market_series = pd.to_numeric(df_sorted.get("Close"), errors="coerce") if "Close" in df_sorted.columns else price_series
    market_series = market_series.fillna(method="ffill").fillna(method="bfill")

    base_date_norm = pd.to_datetime(base_date).normalize()
    latest_data_date = pd.to_datetime(df_sorted.index[-1]).normalize()
    if latest_data_date >= base_date_norm and len(market_series) > 1:
        market_prev = float(market_series.iloc[-2])
    else:
        market_prev = float(market_series.iloc[-1]) if not market_series.empty else None

    market_latest = float(market_series.iloc[-1]) if not market_series.empty else None
    nav_latest: Optional[float] = None
    price_deviation: Optional[float] = None
    daily_pct = 0.0

    country_lower = (country_code or "").strip().lower()
    is_kor_market = country_lower in {"kr", "kor"}
    snapshot_entry = realtime_inav_snapshot.get(ticker_upper) if realtime_inav_snapshot else None

    if is_kor_market and snapshot_entry:
        price_candidate = snapshot_entry.get("nowVal")
        if isinstance(price_candidate, (int, float)) and price_candidate > 0:
            market_latest = float(price_candidate)

        nav_candidate = snapshot_entry.get("nav")
        if isinstance(nav_candidate, (int, float)) and nav_candidate > 0:
            nav_latest = float(nav_candidate)

        deviation_raw = snapshot_entry.get("deviation")
        if isinstance(deviation_raw, (int, float)):
            price_deviation = round(float(deviation_raw), 2)

    # 일간 변동률 계산: 개장 중이거나 실시간 데이터가 있을 때만 계산
    # 개장 전(자정~09:00) 또는 휴장일에는 0으로 표시
    should_calculate_daily_change = False
    if is_kor_market:
        # 한국 시장: 실시간 스냅샷이 있으면 개장 중으로 간주
        if snapshot_entry:
            should_calculate_daily_change = True
        else:
            # 실시간 데이터가 없으면 시장 개장 여부 확인
            from logic.recommend.schedule import is_market_open

            should_calculate_daily_change = is_market_open(country_code)
    else:
        # 해외 시장: 일단 계산 (추후 확장 가능)
        should_calculate_daily_change = True

    if should_calculate_daily_change and market_latest and market_prev and market_prev > 0:
        try:
            daily_pct = ((market_latest / market_prev) - 1.0) * 100
        except ZeroDivisionError:
            daily_pct = 0.0

    if is_kor_market and not snapshot_entry:
        nav_latest = None
        price_deviation = None

    # 데이터 충분성 검증: MA 타입별 이상적인 데이터 요구량
    if config.ENABLE_DATA_SUFFICIENCY_CHECK:
        ma_type_upper = (ma_type or "SMA").upper()
        if ma_type_upper == "TEMA":
            ideal_multiplier = 3.0
        elif ma_type_upper in {"HMA", "EMA", "DEMA"}:
            ideal_multiplier = 2.0
        else:  # SMA, WMA 등
            ideal_multiplier = 1.0

        ideal_data_required = int(ma_period * ideal_multiplier)

        # 데이터가 이상적인 양보다 적으면 완화된 기준 적용 (신규 상장 ETF 대응)
        if len(price_series) < ideal_data_required:
            # 완화된 기준: multiplier의 절반 (최소 1배)
            relaxed_multiplier = max(ideal_multiplier / 2.0, 1.0)
            min_required_data = int(ma_period * relaxed_multiplier)
            logger.info(
                f"[{ticker_upper}] 데이터 부족으로 완화된 기준 적용 (보유: {len(price_series)}개, "
                f"이상: {ideal_data_required}개, 최소: {min_required_data}개, MA타입: {ma_type_upper})"
            )
        else:
            # 충분한 데이터가 있으면 이상적인 기준 적용
            min_required_data = ideal_data_required

        if len(price_series) < min_required_data:
            logger.warning(
                f"[{ticker_upper}] 데이터 부족으로 제외 (보유: {len(price_series)}개, 필요: {min_required_data}개 이상, MA기간: {ma_period})"
            )
            return None

    moving_average = calculate_moving_average(price_series, ma_period, ma_type)
    ma_score_series = calculate_ma_score(price_series, moving_average)
    score_value = float(ma_score_series.iloc[-1]) if not ma_score_series.empty else 0.0
    consecutive_buy_days = get_buy_signal_streak(score_value, ma_score_series, min_buy_score)

    rsi_score = calculate_rsi_for_ticker(price_series)
    if rsi_score == 0.0 and len(price_series) < 15:
        logger.warning(f"[RSI] {ticker_upper} 데이터 부족: {len(price_series)}개 (최소 15개 필요)")

    recent_prices = market_series.tail(63)
    trend_prices = [round(float(val), 6) for val in recent_prices.tolist()] if not recent_prices.empty else []

    drawdown_from_high_pct = 0.0
    if isinstance(price_series, pd.Series):
        price_valid = price_series.dropna()
        if not price_valid.empty:
            try:
                latest_price = float(market_latest) if market_latest is not None else float(price_valid.iloc[-1])
                highest_price = float(price_valid.max())
            except (TypeError, ValueError):
                latest_price = None
                highest_price = None
            if latest_price is not None and highest_price and highest_price > 0:
                drawdown_from_high_pct = round(((latest_price / highest_price) - 1.0) * 100, 2)

    ticker_data = {
        "price": market_latest,
        "nav_price": nav_latest,
        "prev_close": market_prev if market_prev is not None else market_latest,
        "daily_pct": round(daily_pct, 2),
        "close": price_series,
        "s1": moving_average.iloc[-1] if not moving_average.empty else None,
        "s2": None,
        "score": score_value,
        "rsi_score": rsi_score,
        "filter": consecutive_buy_days,
        "ret_1w": _compute_trailing_return(price_series, 5),
        "ret_2w": _compute_trailing_return(price_series, 10),
        "ret_1m": _compute_trailing_return(price_series, 21),
        "ret_3m": _compute_trailing_return(price_series, 63),
        "trend_prices": trend_prices,
        "price_deviation": price_deviation,
        "ma_period": ma_period,
        "drawdown_from_high": drawdown_from_high_pct,
    }
    return ticker_data


def _fetch_trades_for_date(account_id: str, base_date: pd.Timestamp) -> List[Dict[str, Any]]:
    """Retrieve trades executed TODAY (actual current date), not base_date.

    This ensures that before market open, we don't show previous day's trades as SOLD.
    Only trades actually executed today should be marked as SOLD.
    """

    db = get_db_connection()
    if db is None:
        return []

    # 현재 날짜 기준으로 오늘 체결된 거래만 조회 (개장 전에는 빈 리스트 반환)
    now = datetime.now()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    account_norm = (account_id or "").strip().lower()

    cursor = db.trades.find(
        {
            "account": account_norm,
            "deleted_at": {"$exists": False},
            "executed_at": {"$gte": start, "$lt": end},
        },
        projection={"ticker": 1, "action": 1, "name": 1, "_id": 0},
    )

    trades: List[Dict[str, Any]] = []
    for doc in cursor:
        trades.append(
            {
                "ticker": str(doc.get("ticker") or "").upper(),
                "action": str(doc.get("action") or "").upper(),
                "name": str(doc.get("name") or ""),
            }
        )
    return trades


def generate_account_recommendation_report(account_id: str, date_str: Optional[str] = None) -> RecommendationReport:
    """계정 단위 추천 종목 리스트를 반환합니다."""
    if not account_id:
        raise ValueError("account_id 인자가 필요합니다.")
    account_id = account_id.strip().lower()

    base_date = _resolve_base_date(account_id, date_str)

    try:
        strategy_rules = get_strategy_rules(account_id)
        account_settings = get_account_settings(account_id)
    except AccountSettingsError as exc:
        raise ValueError(str(exc)) from exc

    country_code = account_settings.get("country_code")

    ma_period = int(strategy_rules.ma_period)
    portfolio_topn = int(strategy_rules.portfolio_topn)
    ma_type = str(strategy_rules.ma_type)

    strategy_cfg = account_settings.get("strategy", {}) or {}
    if not isinstance(strategy_cfg, dict):
        strategy_cfg = {}

    strategy_tuning = resolve_strategy_params(strategy_cfg)
    min_buy_score = float(strategy_rules.min_buy_score)

    # 검증은 get_account_strategy_sections에서 이미 완료됨 - 바로 사용
    max_per_category = config.MAX_PER_CATEGORY
    rsi_sell_threshold = int(strategy_tuning["OVERBOUGHT_SELL_THRESHOLD"])

    # ETF 목록 가져오기
    etf_universe = get_etfs(country_code) or []
    logger.info(
        "[%s] 추천 Universe 로딩 완료: %d개 종목 데이터 준비",
        account_id.upper(),
        len(etf_universe),
    )
    raw_full_meta = _load_full_etf_meta(country_code)
    full_meta_map: Dict[str, Dict[str, Any]] = {}
    for ticker, meta in raw_full_meta.items():
        norm = str(ticker or "").strip().upper()
        if not norm:
            continue
        entry = dict(meta)
        entry["ticker"] = norm
        entry.setdefault("name", norm)
        if "category" not in entry or not entry["category"]:
            raise ValueError(f"종목 {norm}의 카테고리가 없습니다. 모든 종목은 카테고리가 있어야 합니다.")
        full_meta_map[norm] = entry

    disabled_tickers = {str(stock.get("ticker") or "").strip().upper() for stock in etf_universe if not bool(stock.get("recommend_enabled", True))}
    pairs: List[Tuple[str, str]] = []
    pair_seen: Set[str] = set()
    for stock in etf_universe:
        ticker_value = stock.get("ticker")
        if not ticker_value:
            continue
        ticker_upper = str(ticker_value).strip().upper()
        name_value = stock.get("name") or ticker_upper
        pairs.append((ticker_upper, name_value))
        pair_seen.add(ticker_upper)

    # 실제 포트폴리오 데이터 준비
    holdings: Dict[str, Dict[str, float]] = {}
    try:
        # 현재 미매도 포지션만 조회
        open_positions = list_open_positions(account_id)
        logger.info(
            "[%s] list_open_positions 결과: %d개 - %s",
            account_id.upper(),
            len(open_positions),
            ", ".join([p.get("ticker", "") for p in open_positions]) if open_positions else "(없음)",
        )
        if open_positions:
            for position in open_positions:
                ticker = (position.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                exec_at = position.get("executed_at")
                buy_date = None
                if exec_at is not None:
                    try:
                        buy_date = pd.to_datetime(exec_at).strftime("%Y-%m-%d")
                    except Exception:
                        buy_date = None
                holdings[ticker] = {
                    "buy_date": buy_date,
                }

        # holdings가 비어있어도 신규 계좌의 첫 실행일 수 있으므로 빈 상태로 계속 진행
        if not holdings:
            logger.warning(f"[{account_id.upper()}] 보유 종목이 없어 빈 포트폴리오로 추천을 생성합니다.")

        # 종목명과 티커를 함께 표시
        holdings_display = []
        for ticker in sorted(holdings.keys()):
            meta_entry = full_meta_map.get(ticker) or {}
            name = meta_entry.get("name") or ticker
            holdings_display.append(f"{name}({ticker})")

        logger.info(
            "[%s] 계산된 holdings: %d개 종목 - %s",
            account_id.upper(),
            len(holdings),
            ", ".join(holdings_display) if holdings_display else "(없음)",
        )
    except Exception as e:
        logger.error("포트폴리오 데이터 조회 실패: %s", e)
        holdings = {}

    # 연속 보유 정보 계산
    consecutive_holding_info = calculate_consecutive_holding_info(list(holdings.keys()), account_id, base_date.to_pydatetime())

    # 현재 자산/현금 정보 (임시값 - 실제 계산 필요)
    current_equity = 100_000_000  # 임시값
    total_cash = 100_000_000  # 임시값

    # 당일 거래 내역 확보 (SOLD 표시용)
    trades_today = _fetch_trades_for_date(account_id, base_date)

    rep_ticker_set = {str(stock.get("ticker") or "").strip().upper() for stock in etf_universe if stock.get("ticker")}

    # 보유/거래 티커를 pairs에만 포함
    def _ensure_additional_pair(ticker: Any) -> None:
        text = str(ticker or "").strip().upper()
        if not text:
            return
        if text not in pair_seen:
            meta_entry = full_meta_map.get(text, {})
            name_value = meta_entry.get("name") or text
            pairs.append((text, name_value))
            pair_seen.add(text)

    for held_ticker in holdings.keys():
        _ensure_additional_pair(held_ticker)
    for trade_entry in trades_today:
        _ensure_additional_pair(trade_entry.get("ticker"))

    # 각 티커의 현재 데이터 준비 (실제 OHLCV 데이터 사용)
    tickers_all: List[str] = []
    tickers_seen: Set[str] = set()

    def _append_prefetch_ticker(raw: Any) -> None:
        text = str(raw or "").strip().upper()
        if not text or text in tickers_seen:
            return
        tickers_seen.add(text)
        tickers_all.append(text)

    for stock in etf_universe:
        _append_prefetch_ticker(stock.get("ticker"))
    for ticker_key in holdings.keys():
        _append_prefetch_ticker(ticker_key)
    for trade_entry in trades_today:
        _append_prefetch_ticker(trade_entry.get("ticker"))

    prefetched_data: Dict[str, pd.DataFrame] = {}
    months_back = max(12, ma_period)
    warmup_days = int(max(ma_period, 1) * 1.5)

    common_settings = load_common_settings()
    cache_seed_raw = (common_settings or {}).get("CACHE_START_DATE")
    prefetch_start_dt = base_date - pd.DateOffset(months=months_back)
    if cache_seed_raw:
        try:
            cache_seed_dt = pd.to_datetime(cache_seed_raw).normalize()
            if cache_seed_dt > prefetch_start_dt:
                prefetch_start_dt = cache_seed_dt
        except Exception:
            pass
    start_date = prefetch_start_dt.strftime("%Y-%m-%d")
    end_date = base_date.strftime("%Y-%m-%d")
    primary_tickers = [ticker for ticker in tickers_all if ticker in rep_ticker_set]
    extra_tickers = [ticker for ticker in tickers_all if ticker not in rep_ticker_set]

    logger.info(
        "[%s] 가격 데이터 로딩 시작 (기간 %s~%s, 대상 %d개)",
        account_id.upper(),
        start_date,
        end_date,
        len(tickers_all),
    )
    fetch_start = time.perf_counter()
    prefetched_data: Dict[str, pd.DataFrame] = {}

    if primary_tickers:
        pref_primary, missing_primary = prepare_price_data(
            tickers=primary_tickers,
            country=country_code,
            start_date=start_date,
            end_date=end_date,
            warmup_days=warmup_days,
            allow_remote_fetch=True,  # 캐시가 없으면 원격 조회 허용 (1,2,3주 수익률 계산을 위해 필요)
        )
        if missing_primary:
            raise MissingPriceDataError(
                country=country_code,
                start_date=start_date,
                end_date=end_date,
                tickers=missing_primary,
            )
        prefetched_data.update(pref_primary)

    if extra_tickers:
        pref_extra, missing_extra = prepare_price_data(
            tickers=extra_tickers,
            country=country_code,
            start_date=start_date,
            end_date=end_date,
            warmup_days=warmup_days,
            allow_remote_fetch=True,
        )
        if missing_extra:
            raise MissingPriceDataError(
                country=country_code,
                start_date=start_date,
                end_date=end_date,
                tickers=missing_extra,
            )
        prefetched_data.update(pref_extra)
    logger.info(
        "[%s] 가격 데이터 로딩 완료 (%.1fs)",
        account_id.upper(),
        time.perf_counter() - fetch_start,
    )

    data_by_tkr: Dict[str, Dict[str, Any]] = {}
    country_lower = (country_code or "").strip().lower()
    is_kor_market = country_lower in {"kr", "kor"}
    realtime_inav_snapshot: Dict[str, Dict[str, float]] = {}
    snapshot_targets = (
        [t for t in tickers_all if t]
        if tickers_all
        else [str(stock.get("ticker") or "").strip().upper() for stock in etf_universe if stock.get("ticker")]
    )
    # 실시간 스냅샷은 개장 시간에만 조회 (개장 전/휴장일에는 일간 변동률 0 표시)
    if is_kor_market and snapshot_targets:
        from logic.recommend.schedule import is_market_open

        if is_market_open(country_code):
            try:
                realtime_inav_snapshot = fetch_naver_etf_inav_snapshot(snapshot_targets)
            except Exception as exc:
                logger.warning("[KOR] 네이버 iNAV 스냅샷 조회 실패: %s", exc)
                realtime_inav_snapshot = {}

    missing_data_tickers: List[str] = []
    missing_logged: Set[str] = set()
    for ticker in sorted(tickers_seen):
        df = _fetch_dataframe(
            ticker,
            country=country_code,
            ma_period=ma_period,
            base_date=base_date,
            prefetched_data=prefetched_data,
            data_window=(start_date, end_date),
        )
        if df is None or df.empty:
            missing_data_tickers.append(ticker)
            continue

        entry = _build_ticker_timeseries_entry(
            ticker=ticker,
            df=df,
            country_code=country_code,
            base_date=base_date,
            ma_period=ma_period,
            ma_type=ma_type,
            min_buy_score=min_buy_score,
            realtime_inav_snapshot=realtime_inav_snapshot if is_kor_market else None,
        )
        if entry is None:
            missing_data_tickers.append(ticker)
            continue

        data_by_tkr[ticker] = entry

    if missing_data_tickers:
        extra_missing = set(missing_data_tickers) - missing_logged
        if extra_missing:
            logger.warning(
                "[%s] 분석 중 추가로 제외된 종목: %s",
                account_id.upper(),
                ", ".join(sorted(extra_missing)),
            )
        missing_logged.update(missing_data_tickers)

    # 쿨다운 정보 계산
    trade_cooldown_info = calculate_trade_cooldown_info(
        sorted(tickers_seen),
        account_id,
        base_date.to_pydatetime(),
        country_code=country_code,
    )

    # generate_daily_recommendations_for_portfolio 호출
    etf_meta_map: Dict[str, Dict[str, Any]] = {}
    for stock in etf_universe:
        ticker_value = stock.get("ticker")
        ticker_upper = str(ticker_value or "").strip().upper()
        if not ticker_upper:
            continue
        entry = dict(stock)
        entry["ticker"] = ticker_upper
        entry.setdefault("name", ticker_upper)
        category = stock.get("category") or ""
        if not str(category).strip():
            raise ValueError(f"종목 {ticker_upper}의 카테고리가 없습니다. 모든 종목은 카테고리가 있어야 합니다.")
        entry.setdefault("category", category)
        etf_meta_map[ticker_upper] = entry
    for ticker_upper, meta in full_meta_map.items():
        if ticker_upper not in etf_meta_map:
            etf_meta_map[ticker_upper] = dict(meta)

    try:
        from strategies.maps import safe_generate_daily_recommendations_for_portfolio

        decision_start = time.perf_counter()
        actual_cooldown_days = int(strategy_tuning["COOLDOWN_DAYS"])
        logger.info(
            "[%s] 추천 계산 시작 (보유 %d개, 후보 %d개, cooldown_days=%d)",
            account_id.upper(),
            len(holdings),
            len(data_by_tkr),
            actual_cooldown_days,
        )
        decisions = safe_generate_daily_recommendations_for_portfolio(
            account_id=account_id,
            country_code=country_code,
            base_date=base_date,
            strategy_rules=strategy_rules,
            data_by_tkr=data_by_tkr,
            holdings=holdings,
            etf_meta=etf_meta_map,
            full_etf_meta=full_meta_map,
            current_equity=current_equity,
            total_cash=total_cash,
            pairs=pairs,
            consecutive_holding_info=consecutive_holding_info,
            trade_cooldown_info=trade_cooldown_info,
            cooldown_days=actual_cooldown_days,
            rsi_sell_threshold=rsi_sell_threshold,
        )
        logger.info(
            "[%s] 추천 계산 완료 (%.1fs, 결과 %d개)",
            account_id.upper(),
            time.perf_counter() - decision_start,
            len(decisions) if isinstance(decisions, list) else -1,
        )
    except Exception as exc:
        logger.error("generate_daily_recommendations_for_portfolio 실행 중 오류: %s", exc)
        return []

    # 당일 SELL 트레이드를 결과에 추가하여 SOLD 상태로 노출
    sold_entries: List[Dict[str, Any]] = []
    buy_traded_today: set[str] = set()
    sell_traded_today: set[str] = set()

    # 먼저 SELL 종목 수집
    for trade in trades_today:
        action = (trade.get("action") or "").strip().upper()
        ticker = (trade.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        if action == "SELL":
            sell_traded_today.add(ticker)

    # 이제 거래 처리
    for trade in trades_today:
        action = (trade.get("action") or "").strip().upper()
        ticker = (trade.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        if action == "SELL":
            # 매도 거래가 있으면 SOLD 처리 (부분 매도 여부는 보유 수량으로 판단)
            existing = next((d for d in decisions if d.get("tkr") == ticker), None)
            if existing:
                # 원래 상태 저장 (SELL_RSI 판단용)
                original_state = existing.get("state")
                existing["original_state"] = original_state
                existing["state"] = "SOLD"
                if existing.get("row"):
                    existing["row"][4] = "SOLD"
                    # RSI 과매수 조건 확인하여 메시지 추가
                    rsi_score = existing.get("rsi_score", 0.0)
                    base_msg = DECISION_MESSAGES["SOLD"]
                    if rsi_score >= rsi_sell_threshold:
                        existing["row"][-1] = f"{base_msg} | RSI 과매수 (RSI점수: {rsi_score:.1f})"
                    else:
                        existing["row"][-1] = base_msg
                existing["buy_signal"] = False
                continue

            name = trade.get("name") or ticker
            ticker_data = data_by_tkr.get(ticker, {})
            if not ticker_data:
                meta_info = next(
                    (stock for stock in etf_universe if stock.get("ticker", "").upper() == ticker),
                    None,
                )
                if not meta_info:
                    meta_info = full_meta_map.get(ticker) or full_meta_map.get(ticker.upper())
                if meta_info:
                    name = meta_info.get("name") or name
                else:
                    logger.warning("SOLD 종목 메타데이터 없음: %s", ticker)
                    name = ticker

            price_val = ticker_data.get("price", 0.0)
            daily_pct_val = (
                ticker_data.get("daily_pct", 0.0)
                if "daily_pct" in ticker_data
                else (
                    ((ticker_data.get("price", 0.0) / ticker_data.get("prev_close", 1.0)) - 1.0) * 100
                    if ticker_data.get("prev_close", 0.0) > 0
                    else 0.0
                )
            )
            score_val = float(ticker_data.get("score", 0.0) or 0.0)

            sold_entries.append(
                {
                    "state": "SOLD",
                    "tkr": ticker,
                    "score": score_val,
                    "buy_signal": False,
                    "row": [
                        0,
                        ticker,
                        name,
                        "-",
                        "SOLD",
                        "-",
                        price_val,
                        daily_pct_val,
                        0,
                        0.0,
                        0.0,
                        0.0,
                        "-",
                        score_val,
                        "-",
                        DECISION_MESSAGES["SOLD"],
                    ],
                }
            )

        elif action == "BUY":
            # 당일 SELL이 없는 경우만 BUY로 추가
            if ticker not in sell_traded_today:
                buy_traded_today.add(ticker)

        else:
            continue

    # 결과 포맷팅
    disabled_note = DECISION_NOTES.get("NO_RECOMMEND", "추천 제외")
    held_category_names: Set[str] = set()
    for held_ticker in holdings.keys():
        meta_info = etf_meta_map.get(held_ticker) or {}
        category_value = meta_info.get("category")
        if category_value and not is_category_exception(category_value):
            held_category_names.add(str(category_value).strip())
    results = []
    for decision in decisions:
        ticker = decision["tkr"]
        raw_state = decision["state"]
        phrase = decision["row"][-1] if decision["row"] else ""

        is_currently_held = ticker in holdings

        state = raw_state
        if state in {"BUY", "BUY_REPLACE"}:
            score_val_final = decision.get("score", float("nan"))
            if pd.isna(score_val_final) or score_val_final <= min_buy_score:
                state = "WAIT"
                if decision.get("row"):
                    decision["row"][4] = "WAIT"
                    decision["row"][-1] = _format_min_score_phrase(score_val_final, min_buy_score)
                    phrase = decision["row"][-1]
                decision["buy_signal"] = False
        if is_currently_held and raw_state in {"WAIT"}:
            state = "HOLD"

        new_buy_phrase = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")

        if new_buy_phrase in str(phrase):
            state = "BUY"

        phrase = _format_sell_replace_phrase(phrase, etf_meta=etf_meta_map)

        meta_info = etf_meta_map.get(ticker) or {}
        name = meta_info.get("name", ticker)
        category = meta_info.get("category")
        if not category:
            raise ValueError(f"종목 {ticker}의 카테고리가 없습니다. 모든 종목은 카테고리가 있어야 합니다.")
        ticker_upper = str(ticker).upper()
        recommend_enabled = ticker_upper not in disabled_tickers

        # 보유일 계산 (현재 날짜 기준)
        base_norm = base_date.normalize()
        holding_days_val = 0
        latest_buy_date_norm: Optional[pd.Timestamp] = None
        if ticker in holdings:
            current_date = pd.Timestamp.now().normalize()
            raw_buy_date = consecutive_holding_info.get(ticker, {}).get("buy_date")
            if raw_buy_date:
                buy_timestamp = pd.to_datetime(raw_buy_date).normalize()
                if pd.notna(buy_timestamp):
                    latest_buy_date_norm = buy_timestamp
                    if buy_timestamp <= current_date:
                        holding_days_val = count_trading_days(
                            country_code,
                            buy_timestamp,
                            current_date,
                        )
        elif ticker in consecutive_holding_info:
            buy_date = consecutive_holding_info[ticker].get("buy_date")
            if buy_date:
                buy_timestamp = pd.to_datetime(buy_date).normalize()
                if pd.notna(buy_timestamp) and buy_timestamp <= base_norm:
                    latest_buy_date_norm = buy_timestamp
                    holding_days_val = count_trading_days(
                        country_code,
                        buy_timestamp,
                        base_norm,
                    )

        if latest_buy_date_norm is None:
            fallback_buy_date = holdings.get(ticker, {}).get("buy_date") if ticker in holdings else None
            fallback_norm = _normalize_buy_date(fallback_buy_date)
            if fallback_norm is not None:
                latest_buy_date_norm = fallback_norm

        # 상태 및 문구 재정의
        bought_today = False
        if latest_buy_date_norm is not None and latest_buy_date_norm >= base_norm:
            bought_today = True

        # 당일 매수 체결된 종목 처리
        if ticker in buy_traded_today:
            # HOLD_CORE는 유지하고 항상 "🔒 핵심 보유" 표시
            if state == "HOLD_CORE":
                phrase = DECISION_MESSAGES.get("HOLD_CORE", "🔒 핵심 보유")
            else:
                state = "HOLD"
                # RSI 과매수 조건 확인하여 메시지 추가
                rsi_score_val = decision.get("rsi_score", 0.0)
                base_msg = DECISION_MESSAGES.get("NEWLY_ADDED", "🆕 신규 편입")
                if rsi_score_val >= rsi_sell_threshold:
                    phrase = f"{base_msg} | RSI 과매수 (RSI점수: {rsi_score_val:.1f})"
                else:
                    phrase = base_msg
            if holding_days_val == 0:
                holding_days_val = 1
        # 추천에 따라 오늘 신규 매수해야 할 종목
        elif state in {"BUY", "BUY_REPLACE"}:
            phrase_str = str(phrase)

            if state == "BUY_REPLACE":
                replacement_note = phrase_str
                phrase = _join_phrase_parts(DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수"), replacement_note)
            elif phrase_str:
                phrase = phrase_str
            else:
                phrase = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")
            if holding_days_val == 0:
                holding_days_val = 1
        # 이미 보유 중인 종목이 오늘 신규 편입된 경우
        elif is_currently_held and bought_today:
            # HOLD_CORE는 유지하고 항상 "🔒 핵심 보유" 표시
            if state == "HOLD_CORE":
                phrase = DECISION_MESSAGES.get("HOLD_CORE", "🔒 핵심 보유")
            else:
                state = "HOLD"
                # RSI 과매수 조건 확인하여 메시지 추가
                rsi_score_val = decision.get("rsi_score", 0.0)
                base_msg = DECISION_MESSAGES.get("NEWLY_ADDED", "🆕 신규 편입")
                if rsi_score_val >= rsi_sell_threshold:
                    phrase = f"{base_msg} | RSI 과매수 (RSI점수: {rsi_score_val:.1f})"
                else:
                    phrase = base_msg
            if holding_days_val == 0:
                holding_days_val = 1

        ticker_data = data_by_tkr.get(ticker, {})
        price_val = ticker_data.get("price", 0.0)
        daily_pct_val = (
            ticker_data.get("daily_pct", 0.0)
            if "daily_pct" in ticker_data
            else (
                ((ticker_data.get("price", 0.0) / ticker_data.get("prev_close", 1.0)) - 1.0) * 100 if ticker_data.get("prev_close", 0.0) > 0 else 0.0
            )
        )
        score_val = decision.get("score", 0.0)

        evaluation_pct_val: float = 0.0
        if holding_days_val and holding_days_val > 0 and is_currently_held:
            buy_date_raw = consecutive_holding_info.get(ticker, {}).get("buy_date")
            if not buy_date_raw:
                buy_date_raw = holdings.get(ticker, {}).get("buy_date")

            buy_date_norm = _normalize_buy_date(buy_date_raw)
            if buy_date_norm is None and bought_today:
                buy_date_norm = base_date.normalize()

            buy_price = _resolve_buy_price(
                ticker_data,
                buy_date_norm,
                fallback_price=float(price_val) if price_val else None,
            )

            if buy_price and buy_price > 0 and price_val:
                evaluation_pct_val = round(((float(price_val) / buy_price) - 1.0) * 100, 2)

        ret_1w = ticker_data.get("ret_1w", 0.0)
        ret_2w = ticker_data.get("ret_2w", 0.0)
        ret_1m = ticker_data.get("ret_1m", 0.0)
        ret_3m = ticker_data.get("ret_3m", 0.0)

        filter_days = decision.get("filter")
        if filter_days is None:
            filter_days_row = decision.get("row") or []
            if len(filter_days_row) >= 16:
                try:
                    filter_days = int(str(filter_days_row[15]).replace("일", "")) if filter_days_row[15] not in ("-", None) else 0
                except Exception:
                    filter_days = 0
            else:
                filter_days = 0

        streak_val = int(filter_days)

        if not recommend_enabled:
            if state in {"BUY", "BUY_REPLACE"}:
                state = "WAIT"
            phrase = disabled_note

        rsi_score_val = decision.get("rsi_score", 0.0)

        result_entry = {
            "rank": len(results) + 1,
            "ticker": ticker,
            "name": name,
            "category": category,
            "state": state,
            "price": price_val,
            "price_deviation": ticker_data.get("price_deviation"),
            "daily_pct": daily_pct_val,
            "evaluation_pct": evaluation_pct_val,
            "drawdown_from_high": ticker_data.get("drawdown_from_high", 0.0),
            "return_1w": ret_1w,
            "return_2w": ret_2w,
            "return_1m": ret_1m,
            "return_3m": ret_3m,
            "trend_prices": ticker_data.get("trend_prices", []),
            "score": score_val,
            "rsi_score": rsi_score_val,
            "streak": streak_val,
            "base_date": base_date.strftime("%Y-%m-%d"),
            "holding_days": holding_days_val,
            "phrase": phrase,
            "state_order": DECISION_CONFIG.get(state, {}).get("order", 99),
            "recommend_enabled": recommend_enabled,
        }

        results.append(result_entry)

    for entry in results:
        if entry.get("state") in {"BUY", "BUY_REPLACE"}:
            score_val_final = entry.get("score", float("nan"))
            if pd.isna(score_val_final) or score_val_final <= min_buy_score:
                entry["state"] = "WAIT"
                entry["phrase"] = _format_min_score_phrase(score_val_final, min_buy_score)

        if entry.get("state") == "WAIT" and not entry.get("phrase"):
            category_value = str(entry.get("category") or "").strip()
            if category_value and category_value in held_category_names:
                entry["phrase"] = DECISION_NOTES.get("CATEGORY_DUP", "")

    # BUY 종목 생성: 상위 점수의 WAIT 종목들을 BUY로 변경
    wait_items = [
        item
        for item in results
        if item["state"] == "WAIT"
        and item.get("recommend_enabled", True)
        and not pd.isna(item.get("score"))
        and item.get("score", float("nan")) > min_buy_score
    ]
    # MAPS 점수 기반 정렬
    wait_items.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # 카테고리 보유 제한이 있는 경우, 동일 카테고리 수를 체크 (매도 예정 종목 제외)
    from logic.common import should_exclude_from_category_count

    category_counts: Dict[str, int] = {}
    category_counts_normalized: Dict[str, int] = {}
    category_limit = max_per_category if max_per_category and max_per_category > 0 else 1

    # 매도 예정 종목의 카테고리는 제외하고 카운트
    sell_state_set = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}
    for item in results:
        # 매도 예정 종목은 카테고리 카운트에서 제외
        # HOLD + HOLD_CORE + BUY + BUY_REPLACE = 보유/매수 예정 종목
        if not should_exclude_from_category_count(item["state"]) and item["state"] in {"HOLD", "HOLD_CORE", "BUY", "BUY_REPLACE"}:
            category_raw = item.get("category")
            category = str(category_raw or "").strip()
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
            category_key = _normalize_category_value(category_raw)
            if category_key:
                category_counts_normalized[category_key] = category_counts_normalized.get(category_key, 0) + 1

    current_holdings_count = len(holdings)
    sell_state_set = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}
    buy_state_set = {"BUY", "BUY_REPLACE"}
    planned_sell_count = sum(1 for item in results if item["state"] in sell_state_set)
    planned_buy_count = sum(1 for item in results if item["state"] in buy_state_set)

    # SELL_RSI로 매도되는 카테고리 추적 (같은 날 매수 금지)
    # SOLD 상태도 포함 (RSI 과매수로 매도 완료된 경우)
    sell_rsi_categories: Set[str] = set()
    for item in results:
        if item["state"] == "SELL_RSI":
            category = item.get("category")
            if category and not is_category_exception(category):
                sell_rsi_categories.add(category)
                logger.info(f"[PIPELINE SELL_RSI CAT] {item.get('ticker')} SELL_RSI로 '{category}' 카테고리 추가")
        elif item["state"] == "SOLD":
            # SOLD 상태 중 원래 SELL_RSI였거나 RSI 과매수로 매도된 경우
            original_state = item.get("original_state")
            rsi_score = item.get("rsi_score", 0.0)
            if original_state == "SELL_RSI" or rsi_score >= rsi_sell_threshold:
                category = item.get("category")
                if category and not is_category_exception(category):
                    sell_rsi_categories.add(category)
                    logger.info(
                        f"[PIPELINE SOLD RSI CAT] {item.get('ticker')} SOLD(original={original_state}, RSI={rsi_score:.1f})로 '{category}' 카테고리 추가"
                    )

    # BUY 상태 종목 중 SELL_RSI 카테고리에 해당하는 것은 WAIT로 되돌림
    for item in results:
        if item["state"] in {"BUY", "BUY_REPLACE"}:
            category = item.get("category")
            if category and not is_category_exception(category) and category in sell_rsi_categories:
                logger.info(f"[PIPELINE BUY REVERTED] {item.get('ticker')} BUY→WAIT 변경 - '{category}' 카테고리가 SELL_RSI로 매도됨")
                item["state"] = "WAIT"
                item["phrase"] = f"RSI 과매수 매도 카테고리 ({category})"

    # BUY 상태가 변경되었으므로 planned_buy_count 재계산
    planned_buy_count = sum(1 for item in results if item["state"] in buy_state_set)

    projected_holdings = current_holdings_count - planned_sell_count + planned_buy_count
    additional_buy_slots = max(0, portfolio_topn - projected_holdings)

    # logger.info(
    #     f"[PIPELINE] 매수 슬롯 계산: current={current_holdings_count}, sell={planned_sell_count}, buy={planned_buy_count}, projected={projected_holdings}, topn={portfolio_topn}, slots={additional_buy_slots}, wait_items={len(wait_items)}"
    # )

    promoted = 0
    for item in wait_items:
        if promoted >= additional_buy_slots:
            break

        category_raw = item.get("category")
        category = str(category_raw or "").strip()
        category_key = _normalize_category_value(category_raw)

        # SELL_RSI로 매도한 카테고리는 같은 날 매수 금지
        if category and not is_category_exception(category) and category in sell_rsi_categories:
            logger.info(f"[PIPELINE BUY BLOCKED] {item.get('ticker')} 매수 차단 - '{category}' 카테고리가 SELL_RSI로 매도됨")
            continue

        # 카테고리 중복 체크 시, 매도 예정 종목은 제외하고 매수 예정 종목은 포함
        # 같은 카테고리의 매도 예정 종목이 있으면 해당 카테고리 슬롯이 비게 됨
        sell_in_same_category = sum(
            1 for r in results if r["state"] in sell_state_set and _normalize_category_value(r.get("category")) == category_key
        )
        # BUY_REPLACE로 이미 추가된 같은 카테고리 종목도 카운트
        buy_replace_in_same_category = sum(
            1 for r in results if r["state"] == "BUY_REPLACE" and _normalize_category_value(r.get("category")) == category_key
        )
        effective_category_count = category_counts_normalized.get(category_key, 0) - sell_in_same_category + buy_replace_in_same_category

        if category_key and effective_category_count >= category_limit:
            # 카테고리 중복인 경우 BUY로 변경하지 않고 WAIT 상태 유지
            logger.info(f"[PIPELINE BUY BLOCKED] {item.get('ticker')} 매수 차단 - '{category}' 카테고리 중복 (현재 {effective_category_count}개)")
            continue

        # RSI 과매수 종목 매수 차단
        rsi_score = item.get("rsi_score", 0.0)
        if rsi_score >= rsi_sell_threshold:
            logger.info(f"[PIPELINE BUY BLOCKED] {item.get('ticker')} 매수 차단 - RSI 과매수 (RSI점수: {rsi_score:.1f})")
            continue

        item["state"] = "BUY"
        item["phrase"] = DECISION_MESSAGES.get("NEW_BUY", "✅ 신규 매수")
        promoted += 1

        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        if category_key:
            category_counts_normalized[category_key] = category_counts_normalized.get(category_key, 0) + 1

        # 신규 매수로 전환된 종목은 holdings 정보가 없으므로 기본값 추가
        holdings.setdefault(
            item["ticker"],
            {
                "buy_date": base_date.strftime("%Y-%m-%d"),
            },
        )

    sell_state_set = {
        "SELL_TREND",
        "SELL_REPLACE",
        "CUT_STOPLOSS",
    }
    buy_state_set = {"BUY", "BUY_REPLACE"}

    planned_sell_count = sum(1 for item in results if (item.get("state") or "").upper() in sell_state_set)
    planned_buy_count = sum(1 for item in results if (item.get("state") or "").upper() in buy_state_set)
    projected_holdings = current_holdings_count - planned_sell_count + planned_buy_count

    if projected_holdings > portfolio_topn:
        logger.debug(
            "Projected holdings (%d) exceed portfolio_topn(%d); trim logic removed",
            projected_holdings,
            portfolio_topn,
        )

    # rank를 MAPS 점수 순서대로 재설정
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    for i, item in enumerate(results, 1):
        item["rank"] = i

    # 최종 state_order 재계산 및 상태 정렬 (백테스트와 동일한 기준 사용)
    for item in results:
        state_key = (item.get("state") or "").upper()
        item["state_order"] = DECISION_CONFIG.get(state_key, {}).get("order", 99)

    # 공통 정렬 함수 사용
    sort_decisions_by_order_and_score(results)

    # sort 후 rank 재설정
    for i, item in enumerate(results, 1):
        item["rank"] = i

    # 카테고리별 최고 점수만 표시 (교체 매매 제외)
    results = filter_category_duplicates(results, category_key_getter=_normalize_category_value)

    # rank 재설정
    for i, item in enumerate(results, 1):
        item["rank"] = i

    # 계정 설정에 따라 HOLD/HOLD_CORE만 표시
    show_hold_only = bool(
        account_settings.get("show_hold_only")
        or account_settings.get("show_held_only")
        or (isinstance(strategy_cfg, dict) and strategy_cfg.get("SHOW_HOLD_ONLY"))
    )
    if show_hold_only:
        allowed_states = {"HOLD", "HOLD_CORE"}
        results = [item for item in results if (item.get("state") or "").upper() in allowed_states]
        for i, item in enumerate(results, 1):
            item["rank"] = i

    price_header = "현재가"
    for item in results:
        ticker_key = item.get("ticker")
        source_entry = data_by_tkr.get(ticker_key, {}) if ticker_key else {}
        if "nav_price" in source_entry and item.get("nav_price") is None:
            item["nav_price"] = source_entry.get("nav_price")
        if item.get("price") is None and source_entry.get("price") is not None:
            item["price"] = source_entry.get("price")
        if item.get("price_deviation") is None and source_entry.get("price_deviation") is not None:
            item["price_deviation"] = source_entry.get("price_deviation")

    show_deviation = country_lower in {"kr", "kor"}

    detail_headers = [
        "순위",
        "티커",
        "종목명",
        "카테고리",
        "상태",
        "보유일",
        "일간(%)",
        "평가(%)",
        price_header,
    ]
    if show_deviation:
        detail_headers.append("괴리율")
    detail_headers.extend(["1주(%)", "2주(%)", "1달(%)", "3달(%)", "고점대비", "점수", "지속", "문구"])

    detail_rows: List[List[Any]] = []
    for item in results:
        row = [
            item.get("rank", 0),
            item.get("ticker"),
            item.get("name"),
            item.get("category"),
            item.get("state"),
            item.get("holding_days"),
            item.get("daily_pct"),
            item.get("evaluation_pct"),
            item.get("price"),
        ]
        if show_deviation:
            row.append(item.get("price_deviation"))
        row.extend(
            [
                item.get("return_1w"),
                item.get("return_2w"),
                item.get("return_1m"),
                item.get("return_3m"),
                item.get("drawdown_from_high"),
                item.get("score"),
                item.get("streak"),
                item.get("phrase", ""),
            ]
        )
        detail_rows.append(row)

    report = RecommendationReport(
        account_id=account_id,
        country_code=country_code,
        base_date=base_date,
        recommendations=results,
        report_date=datetime.now(),
        summary_data=None,
        header_line=None,
        detail_headers=detail_headers,
        detail_rows=detail_rows,
        detail_extra_lines=None,
        decision_config=DECISION_CONFIG,
    )

    return report


# 하위 호환: 기존 함수명을 그대로 제공
generate_country_recommendation_report = generate_account_recommendation_report


__all__ = [
    "generate_account_recommendation_report",
    "generate_country_recommendation_report",
]


def _normalize_category_value(category: Optional[str]) -> Optional[str]:
    """Normalize category strings for comparison."""
    if category is None:
        return None
    category_str = str(category).strip()
    if not category_str:
        return None
    return category_str.upper()
