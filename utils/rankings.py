from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from config import BUCKET_MAPPING, CACHE_START_DATE, MIN_TRADING_DAYS, TRADING_DAYS_PER_MONTH
from core.strategy.metrics import process_ticker_data
from services.price_service import get_realtime_snapshot
from utils.cache_utils import load_cached_frames_bulk_with_fallback, load_cached_updated_at_bulk_with_fallback
from utils.logger import get_app_logger
from utils.portfolio_io import load_portfolio_master
from utils.settings_loader import AccountSettingsError, get_account_settings
from utils.stock_list_io import get_etfs

ALLOWED_MA_TYPES = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA", "ALMA"]
logger = get_app_logger()


def _calculate_rsi(close_series: pd.Series, period: int = 14) -> float | None:
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if len(series) < period + 1:
        return None

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    if avg_loss.empty or pd.isna(avg_loss.iloc[-1]):
        return None
    if float(avg_loss.iloc[-1]) == 0.0:
        return 100.0
    rs = float(avg_gain.iloc[-1]) / float(avg_loss.iloc[-1])
    return 100.0 - (100.0 / (1.0 + rs))


def get_account_rank_defaults(account_id: str) -> tuple[str, int]:
    settings = get_account_settings(account_id)

    ma_type = str(settings.get("MA_TYPE") or "").strip().upper()
    if not ma_type:
        raise AccountSettingsError(f"'{account_id}' 설정에 필수 항목 'MA_TYPE'가 누락되었습니다.")
    if ma_type not in ALLOWED_MA_TYPES:
        raise AccountSettingsError(f"'{account_id}' 설정의 MA_TYPE이 유효하지 않습니다: {ma_type}")

    ma_months_raw = settings.get("MA_MONTHS")
    if ma_months_raw is None:
        raise AccountSettingsError(f"'{account_id}' 설정에 필수 항목 'MA_MONTHS'가 누락되었습니다.")

    try:
        ma_months = int(ma_months_raw)
    except (TypeError, ValueError) as exc:
        raise AccountSettingsError(f"'{account_id}' 설정의 MA_MONTHS는 정수여야 합니다: {ma_months_raw}") from exc
    if ma_months < 1:
        raise AccountSettingsError(f"'{account_id}' 설정의 MA_MONTHS는 1 이상이어야 합니다: {ma_months}")

    return ma_type, ma_months


def get_rank_months_max() -> int:
    cache_start = pd.to_datetime(CACHE_START_DATE).normalize()
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    month_span = ((today.year - cache_start.year) * 12) + (today.month - cache_start.month) + 1
    return max(1, int(month_span))


def _calc_period_return(close_series: pd.Series, days: int) -> float | None:
    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return None

    current = float(series.iloc[-1])
    if current <= 0:
        return None

    if len(series) > days:
        previous = float(series.iloc[-(days + 1)])
        if previous > 0:
            return (current / previous - 1.0) * 100.0

    if days == 252 and len(series) >= 240:
        previous = float(series.iloc[0])
        if previous > 0:
            return (current / previous - 1.0) * 100.0

    return None


def _extract_price_metrics_from_close_series(close_series: pd.Series | None) -> dict[str, Any]:
    empty_result = {
        "현재가": None,
        "괴리율": None,
        "일간(%)": None,
        "1주(%)": None,
        "2주(%)": None,
        "1달(%)": None,
        "3달(%)": None,
        "6달(%)": None,
        "12달(%)": None,
        "고점대비": None,
        "추세(3달)": [],
        "RSI": None,
    }
    if close_series is None:
        return empty_result

    series = pd.to_numeric(close_series, errors="coerce").dropna()
    if series.empty:
        return empty_result

    current_price = float(series.iloc[-1])
    daily_pct = None
    if len(series) > 1:
        prev_close = float(series.iloc[-2])
        if prev_close > 0:
            daily_pct = ((current_price / prev_close) - 1.0) * 100.0

    max_price = float(series.max()) if not series.empty else 0.0
    drawdown = None
    if max_price > 0:
        drawdown = (current_price / max_price - 1.0) * 100.0

    return {
        "현재가": current_price,
        "일간(%)": daily_pct,
        "1주(%)": _calc_period_return(series, 5),
        "2주(%)": _calc_period_return(series, 10),
        "1달(%)": _calc_period_return(series, 20),
        "3달(%)": _calc_period_return(series, 60),
        "6달(%)": _calc_period_return(series, 126),
        "12달(%)": _calc_period_return(series, 252),
        "고점대비": drawdown,
        "추세(3달)": series.iloc[-60:].astype(float).tolist(),
        "RSI": _calculate_rsi(series),
    }


def _load_realtime_snapshot(country_code: str, tickers: list[str]) -> dict[str, dict[str, float]]:
    """국가별 실시간 현재가/등락률 스냅샷을 로드합니다."""
    if not tickers:
        return {}

    try:
        return get_realtime_snapshot(country_code, tickers)
    except Exception as exc:
        logger.warning("순위용 실시간 스냅샷 조회 실패: %s", exc)
        return {}


def _apply_realtime_overlay(
    price_metrics: dict[str, Any],
    realtime_entry: dict[str, float] | None,
) -> dict[str, Any]:
    """실시간 현재가/일간 등락률이 있으면 캐시 값을 덮어씁니다."""
    if not isinstance(realtime_entry, dict) or not realtime_entry:
        return price_metrics

    updated = dict(price_metrics)

    now_val = realtime_entry.get("nowVal")
    if now_val is not None:
        try:
            updated["현재가"] = float(now_val)
        except (TypeError, ValueError):
            pass

    change_rate = realtime_entry.get("changeRate")
    if change_rate is not None:
        try:
            updated["일간(%)"] = float(change_rate)
        except (TypeError, ValueError):
            pass

    deviation = realtime_entry.get("deviation")
    if deviation is not None:
        try:
            updated["괴리율"] = float(deviation)
        except (TypeError, ValueError):
            pass

    return updated


def _build_effective_close_series(
    cached_df: pd.DataFrame | None,
    realtime_entry: dict[str, float] | None,
) -> pd.Series | None:
    """실시간 가격을 반영한 종가 시리즈를 생성합니다."""
    if cached_df is None or cached_df.empty:
        return None
    if not isinstance(realtime_entry, dict) or not realtime_entry:
        return _extract_close_series(cached_df)

    now_val = realtime_entry.get("nowVal")
    if now_val is None:
        return _extract_close_series(cached_df)

    try:
        realtime_price = float(now_val)
    except (TypeError, ValueError):
        return _extract_close_series(cached_df)

    close_series = _extract_close_series(cached_df)
    if close_series is None or close_series.empty:
        return close_series

    adjusted = close_series.copy()
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    last_index = pd.Timestamp(adjusted.index[-1])
    if last_index.tzinfo is not None:
        last_index = last_index.tz_localize(None)
    last_index = last_index.normalize()

    if last_index < today:
        adjusted.loc[today] = realtime_price
    else:
        adjusted.iloc[-1] = realtime_price
    return adjusted.sort_index()


def _extract_close_series(cached_df: pd.DataFrame | None) -> pd.Series | None:
    if cached_df is None or cached_df.empty:
        return None

    close_col = "Close" if "Close" in cached_df.columns else "close" if "close" in cached_df.columns else None
    if close_col is None:
        return None

    close_series = pd.to_numeric(cached_df[close_col], errors="coerce").dropna()
    if close_series.empty:
        return None
    return close_series.astype(float)


def _normalize_ranking_values(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    normalized = df.copy()

    price_digits = 2 if str(country_code or "").strip().lower() == "au" else 0
    percent_columns = ["괴리율", "일간(%)", "1주(%)", "2주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)", "고점대비"]
    one_decimal_columns = ["점수", "RSI"]

    def _round_if_present(column: str, digits: int) -> None:
        if column not in normalized.columns:
            return
        series = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = series.round(digits)

    _round_if_present("현재가", price_digits)
    _round_if_present("지속", 0)

    for column in percent_columns:
        _round_if_present(column, 2)

    for column in one_decimal_columns:
        _round_if_present(column, 1)

    if "지속" in normalized.columns:
        normalized["지속"] = pd.to_numeric(normalized["지속"], errors="coerce").astype("Int64")

    return normalized


def build_account_rankings(
    account_id: str,
    *,
    ma_type: str,
    ma_months: int,
    realtime_snapshot_override: dict[str, dict[str, float]] | None = None,
    held_tickers_override: set[str] | None = None,
) -> pd.DataFrame:
    settings = get_account_settings(account_id)
    country_code = str(settings.get("country_code") or "").strip().lower()

    etfs = get_etfs(account_id)
    if not etfs:
        return pd.DataFrame()

    tickers = [str(item.get("ticker") or "").strip().upper() for item in etfs if str(item.get("ticker") or "").strip()]
    cached_frames = load_cached_frames_bulk_with_fallback(account_id, tickers)
    realtime_snapshot = (
        realtime_snapshot_override
        if realtime_snapshot_override is not None
        else _load_realtime_snapshot(country_code, tickers)
    )

    held_tickers: set[str] = set()
    if held_tickers_override is not None:
        held_tickers = {str(ticker).strip().upper() for ticker in held_tickers_override if str(ticker or "").strip()}
    else:
        portfolio_master = load_portfolio_master(account_id)
        if portfolio_master:
            held_tickers = {
                str(holding.get("ticker") or "").strip().upper()
                for holding in portfolio_master.get("holdings", [])
                if str(holding.get("ticker") or "").strip()
            }

    ma_days = int(ma_months) * int(TRADING_DAYS_PER_MONTH)
    rows: list[dict[str, Any]] = []

    for etf in etfs:
        ticker = str(etf.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        cached_df = cached_frames.get(ticker)
        realtime_entry = realtime_snapshot.get(ticker)
        effective_close_series = _build_effective_close_series(cached_df, realtime_entry)
        price_metrics = _extract_price_metrics_from_close_series(effective_close_series)
        price_metrics = _apply_realtime_overlay(price_metrics, realtime_entry)
        score_value = None
        streak_value: int | None = None

        if effective_close_series is not None and not effective_close_series.empty:
            processed = process_ticker_data(
                ticker,
                cached_df,
                ma_days=ma_days,
                precomputed_entry={"close": effective_close_series, "open": effective_close_series},
                ma_type=ma_type,
                enable_data_sufficiency_check=False,
            )
            if processed is not None:
                score_series = processed.get("ma_score")
                streak_series = processed.get("buy_signal_days")
                if isinstance(score_series, pd.Series) and not score_series.empty:
                    score_raw = score_series.iloc[-1]
                    if pd.notna(score_raw):
                        score_value = float(score_raw)
                if isinstance(streak_series, pd.Series) and not streak_series.empty:
                    streak_raw = streak_series.iloc[-1]
                    if pd.notna(streak_raw):
                        streak_value = int(streak_raw)
        elif effective_close_series is not None and len(effective_close_series.index) >= MIN_TRADING_DAYS:
            pass

        rows.append(
            {
                "버킷": BUCKET_MAPPING.get(int(etf.get("bucket") or 0), str(etf.get("bucket") or "")),
                "bucket": int(etf.get("bucket") or 0),
                "티커": ticker,
                "종목명": etf.get("name", ""),
                "상장일": etf.get("listing_date", "-"),
                "점수": score_value,
                "지속": streak_value,
                "보유": "보유" if ticker in held_tickers else "",
                **price_metrics,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    data_updated_at: datetime | None = None
    realtime_active = bool(realtime_snapshot)
    if realtime_active:
        data_updated_at = datetime.now()
    else:
        cache_updated_map = load_cached_updated_at_bulk_with_fallback(account_id, tickers)
        if cache_updated_map:
            data_updated_at = max(cache_updated_map.values())

    def _sort_key(row: pd.Series) -> tuple[int, float, str]:
        score = row.get("점수")
        if score is None or pd.isna(score):
            return (1, float("-inf"), str(row.get("티커", "")))
        return (0, float(score), str(row.get("티커", "")))

    sort_values = df.apply(_sort_key, axis=1, result_type="expand")
    sort_values.columns = ["_missing_score", "_score_value", "_ticker_sort"]
    df = pd.concat([df, sort_values], axis=1)
    df = df.sort_values(
        by=["_missing_score", "_score_value", "_ticker_sort"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    df.insert(0, "#", range(1, len(df) + 1))
    df = df.drop(columns=["_missing_score", "_score_value", "_ticker_sort"])
    df = _normalize_ranking_values(df, country_code)
    df.attrs["realtime_active"] = realtime_active
    if data_updated_at is not None:
        df.attrs["data_updated_at"] = data_updated_at
    return df


__all__ = [
    "ALLOWED_MA_TYPES",
    "build_account_rankings",
    "get_account_rank_defaults",
    "get_rank_months_max",
]
