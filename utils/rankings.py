from __future__ import annotations

from typing import Any

import pandas as pd

from config import BUCKET_MAPPING, CACHE_START_DATE, MIN_TRADING_DAYS, TRADING_DAYS_PER_MONTH
from core.strategy.metrics import process_ticker_data
from utils.cache_utils import load_cached_frames_bulk_with_fallback
from utils.portfolio_io import load_real_holdings_with_recommendations
from utils.settings_loader import AccountSettingsError, get_account_settings
from utils.stock_list_io import get_etfs

ALLOWED_MA_TYPES = ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA", "ALMA"]


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
        strategy_cfg = settings.get("strategy")
        if isinstance(strategy_cfg, dict):
            ma_type = str(strategy_cfg.get("MA_TYPE") or "").strip().upper()
    if not ma_type:
        raise AccountSettingsError(f"'{account_id}' 설정에 필수 항목 'MA_TYPE'가 누락되었습니다.")
    if ma_type not in ALLOWED_MA_TYPES:
        raise AccountSettingsError(f"'{account_id}' 설정의 MA_TYPE이 유효하지 않습니다: {ma_type}")

    ma_months_raw = settings.get("MA_MONTHS")
    if ma_months_raw is None:
        strategy_cfg = settings.get("strategy")
        if isinstance(strategy_cfg, dict):
            ma_months_raw = strategy_cfg.get("MA_MONTHS")
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


def _extract_price_metrics(cached_df: pd.DataFrame | None) -> dict[str, Any]:
    empty_result = {
        "현재가": None,
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
    if cached_df is None or cached_df.empty:
        return empty_result

    close_col = "Close" if "Close" in cached_df.columns else "close"
    if close_col not in cached_df.columns:
        return empty_result

    close_series = pd.to_numeric(cached_df[close_col], errors="coerce").dropna()
    if close_series.empty:
        return empty_result

    current_price = float(close_series.iloc[-1])
    daily_pct = None
    if len(close_series) > 1:
        prev_close = float(close_series.iloc[-2])
        if prev_close > 0:
            daily_pct = ((current_price / prev_close) - 1.0) * 100.0

    max_price = float(close_series.max()) if not close_series.empty else 0.0
    drawdown = None
    if max_price > 0:
        drawdown = (current_price / max_price - 1.0) * 100.0

    return {
        "현재가": current_price,
        "일간(%)": daily_pct,
        "1주(%)": _calc_period_return(close_series, 5),
        "2주(%)": _calc_period_return(close_series, 10),
        "1달(%)": _calc_period_return(close_series, 20),
        "3달(%)": _calc_period_return(close_series, 60),
        "6달(%)": _calc_period_return(close_series, 126),
        "12달(%)": _calc_period_return(close_series, 252),
        "고점대비": drawdown,
        "추세(3달)": close_series.iloc[-60:].astype(float).tolist(),
        "RSI": _calculate_rsi(close_series),
    }


def build_account_rankings(account_id: str, *, ma_type: str, ma_months: int) -> pd.DataFrame:
    etfs = get_etfs(account_id)
    if not etfs:
        return pd.DataFrame()

    tickers = [str(item.get("ticker") or "").strip().upper() for item in etfs if str(item.get("ticker") or "").strip()]
    cached_frames = load_cached_frames_bulk_with_fallback(account_id, tickers)

    held_tickers: set[str] = set()
    holdings_df = load_real_holdings_with_recommendations(account_id)
    if holdings_df is not None and not holdings_df.empty and "티커" in holdings_df.columns:
        held_tickers = {
            str(ticker).strip().upper() for ticker in holdings_df["티커"].tolist() if str(ticker or "").strip()
        }

    ma_days = int(ma_months) * int(TRADING_DAYS_PER_MONTH)
    rows: list[dict[str, Any]] = []

    for etf in etfs:
        ticker = str(etf.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        cached_df = cached_frames.get(ticker)
        price_metrics = _extract_price_metrics(cached_df)
        score_value = None
        streak_value: int | None = None

        if cached_df is not None and not cached_df.empty:
            processed = process_ticker_data(
                ticker,
                cached_df,
                ma_days=ma_days,
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
        elif cached_df is not None and len(cached_df.index) >= MIN_TRADING_DAYS:
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
    return df.drop(columns=["_missing_score", "_score_value", "_ticker_sort"])


__all__ = [
    "ALLOWED_MA_TYPES",
    "build_account_rankings",
    "get_account_rank_defaults",
    "get_rank_months_max",
]
